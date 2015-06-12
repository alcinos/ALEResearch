/**
 * @file   DqnLearner.cpp
 * @author Nicolas Carion
 * @date   Fri Jun  5 16:45:48 2015
 *
 * @brief  Implementation file of the Dqn agent
 *
 *
 */
#include "Dqn.hpp"
#include <ctime>
#include "../../../common/Timer.hpp"

#include <caffe/util/upgrade_proto.hpp> 
DqnLearner::DqnLearner(Environment<Pixel>& env, Parameters* param) : RLLearner<Pixel>(env,param){
    alpha = param->getAlpha();
    lambda = param->getLambda();
    m_playFreq = param->getNumStepsPerAction();
    m_replay_size = 1000000;
    m_replay_memory = replay_memory(m_replay_size);
    
    //caffe::Caffe::SetDevice(0);
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    caffe::Caffe::DeviceQuery();
    
    // Initialize net and solver
    caffe::NetParameter *net_param = new caffe::NetParameter();
    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie("dqn_solver.prototxt", &solver_param);
    caffe::ReadNetParamsFromTextFileOrDie("dqn_net.prototxt",net_param);
    cout<<"Setting the correct number of actions..."<<endl;
    for(int i = 0;i<net_param->layers_size();i++){
        auto lay = net_param->mutable_layers(i);
        if(lay->top_size()>0&&lay->top(0)=="q_values"){
            auto inner = lay->mutable_inner_product_param();
            inner->set_num_output(this->actions.size());
        }
    }
    solver_param.set_allocated_net_param(net_param);
    m_solver.reset(caffe::GetSolver<float>(solver_param));
    m_net = m_solver->net();
    m_frame_input_layer =  boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>( m_net->layer_by_name("frames_input_layer"));
    m_target_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(m_net->layer_by_name("target_input_layer"));
    m_q_values_blob = m_net->blob_by_name("q_values");
    m_Q = vector<double>(18);

}
void DqnLearner::replay_memory::storeFrame(const std::vector<Pixel>& frame)
{
    snappy::Compress(reinterpret_cast<const char*>(frame.data()),frame.size(),&frames[cur_pos]);
}
void DqnLearner::replay_memory::storeReward(float reward)
{
    rewards[cur_pos] = reward;
    cur_pos = (cur_pos + 1) % rewards.size(); //the reward is always stored last, hence we have to increment here
    num_stored = min((int)rewards.size(),num_stored+1);
}
void DqnLearner::replay_memory::storeAction(int action){
    actions[cur_pos] = action;
}
void DqnLearner::replay_memory::storeTermination(bool term){
    terminations[cur_pos] = term;
}
void DqnLearner::learnPolicy(Environment<Pixel>& env){
    m_replay_memory.clear();
    //we recall that the reward associated to an action is the reward obtained when performing it and all its repetition (in case frame skip is not 1)
    double reward = 0, cumReward = 0, prevCumReward = 0;
    struct timeval tvBegin, tvEnd, tvDiff;
	double elapsedTime;

    vector<Pixel> current_frame;
    std::array<vector<Pixel>, m_numFramesPerInput> frame_buffer; //this is where the last frames seen are stored
    int frame_buffer_size = 0; //number of different frames in the buffer
    int frame_buffer_index = 0; //position where the next frame should be written
    
    bool firstActionTaken = false; 
    int totalNumberFrames = 0;
    bool correctlySaved = false; //whether the last action was correctly saved in the replay mem. Incorrect saving occurs when the episode terminates before the next action is chosen.
	for(int episode = 0; totalNumberFrames < totalNumberOfFramesToLearn; episode++){
		//Repeat(for each step of episode) until game is over:
        int currentAction = 0;
        gettimeofday(&tvBegin, NULL);
        int step = -1;
        reward = 0;
		while(!env.isTerminal()){
            step++;
            //get the current frame. We must call this function even if the frame will be skipped
            env.getRawFeatures(current_frame);
            if(step%m_playFreq !=0){
                //don't forget to keep track of the rewards during frame skip
                reward+=env.act(actions[currentAction]);
                //cout<<"skipping with "<<currentAction<<endl;
                continue; //manual frame skip
            }
            //if we reach this point, it means that the last action has not led to termination, we can store tihs information in the replay mem.
            m_replay_memory.storeTermination(false);
            //at this point, reward contain the accumulated reward over the past skipped frames. We have to store it, crediting the last action taken, and reinit it.
            m_replay_memory.storeReward(reward);
            reward = 0;
            //acknowledge that the saving is successfull
            correctlySaved = true;
            
            //store the current frame in the buffer
            std::swap(current_frame,frame_buffer[frame_buffer_index]);
            frame_buffer_size = min(frame_buffer_size+1,m_numFramesPerInput);
            frame_buffer_index = (frame_buffer_index+1) % m_numFramesPerInput;

            //store it also in the replay memory
            m_replay_memory.storeFrame(current_frame);
            if(frame_buffer_size < m_numFramesPerInput){
                //we do not have enough frames to feed the net, we just noop
                reward+=env.act(PLAYER_A_NOOP);
                cout<<"playing a noop"<<endl;
                continue;
            }
            feedNet(frame_buffer,frame_buffer_index);
            updateQValues();
            currentAction = epsilonGreedy(m_Q);
            
            correctlySaved = false;
            firstActionTaken = true;
            //cout<<"playing with "<<currentAction<<endl;
			//Take action, observe reward and next state:
			reward += env.act(actions[currentAction]);
            //cout<<"playing"<<currentAction<<endl;
			cumReward  += reward;
		}
        if(!correctlySaved){
            //the last action led to termination
            m_replay_memory.storeTermination(true);
            m_replay_memory.storeReward(reward);
            reward = 0;
        }
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
		double fps = double(env.getEpisodeFrameNumber())/elapsedTime;
        double fps2 = double(step)/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps,\t %.0f\n", 
               episode + 1, (cumReward-prevCumReward), (double)cumReward/(episode + 1.0), env.getEpisodeFrameNumber(), fps, fps2);
		totalNumberFrames += env.getEpisodeFrameNumber();
		env.reset();
		prevCumReward = cumReward;
	}
}

void DqnLearner::feedNet(std::array<std::vector<Pixel>, m_numFramesPerInput>& input_buffer, int current_buffer_index)
{
    memset(m_input_buff,0,m_numFramesPerInput*m_imageDim*m_imageDim*m_batchSize*sizeof(float));
    int size_written = 0;
    //copy the data in the input blob
    for(int i = 0; i< m_numFramesPerInput; i++){
        int cur_pos = (i+current_buffer_index) % m_numFramesPerInput;
        std::copy(input_buffer[cur_pos].begin(),input_buffer[cur_pos].end(),m_input_buff + size_written);
        size_written += input_buffer[cur_pos].size();
    }
    m_frame_input_layer->Reset(m_input_buff,m_dummy_labels,m_batchSize);
    m_target_input_layer->Reset(m_target_buff,m_dummy_labels,m_batchSize);
    //run forward computation
    m_net->ForwardPrefilled();
}

void DqnLearner::updateQValues()
{
    const float *q_from_net = m_q_values_blob->cpu_data();
    copy(q_from_net,q_from_net+(int)m_Q.size(),m_Q.begin());
}
void DqnLearner::evaluatePolicy(Environment<Pixel> & env)
{
    double reward = 0, cumReward = 0, prevCumReward = 0;
	struct timeval tvBegin, tvEnd, tvDiff;
	double elapsedTime;

    vector<Pixel> current_frame;
    std::array<vector<Pixel>, m_numFramesPerInput> frame_buffer; //this is where the last frames seen are stored
    int frame_buffer_size = 0; //number of different frames in the buffer
    int frame_buffer_index = 0; //position where the next frame should be written
	//Repeat (for each episode):
	for(int episode = 0; episode < numEpisodesEval; episode++){
		//Repeat(for each step of episode) until game is over:
        int currentAction = 0;
        gettimeofday(&tvBegin, NULL);
        int step;
		for(step = 0; !env.isTerminal() && step < episodeLength; step++){
            //get the current frame. We must call this function even if the frame will be skipped
            env.getRawFeatures(current_frame);
            if(step%m_playFreq !=0){
                //don't forget to keep track of the rewards during frame skip
                reward+=env.act(actions[currentAction]);
                //cout<<"skipping with "<<currentAction<<endl;
                continue; //manual frame skip
            }
            //store the current frame in the buffer
            std::swap(current_frame,frame_buffer[frame_buffer_index]);
            frame_buffer_size = min(frame_buffer_size+1,m_numFramesPerInput);
            frame_buffer_index = (frame_buffer_index+1) % m_numFramesPerInput;

            if(frame_buffer_size < m_numFramesPerInput){
                //we do not have enough frames to feed the net, we just noop
                reward+=env.act(PLAYER_A_NOOP);
                cout<<"playing a noop"<<endl;
                continue;
            }
            feedNet(frame_buffer,frame_buffer_index);
            updateQValues();
            currentAction = epsilonGreedy(m_Q);
            //cout<<"playing with "<<currentAction<<endl;
			//Take action, observe reward and next state:
			reward += env.act(actions[currentAction]);
            //cout<<"playing"<<currentAction<<endl;
			cumReward  += reward;
            reward = 0;
            step++;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
		double fps = double(env.getEpisodeFrameNumber())/elapsedTime;
        double fps2 = double(step)/elapsedTime;
		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps,\t %.0f\n", 
               episode + 1, (cumReward-prevCumReward), (double)cumReward/(episode + 1.0), env.getEpisodeFrameNumber(), fps, fps2);

		env.reset();
		prevCumReward = cumReward;
	}
    

}



