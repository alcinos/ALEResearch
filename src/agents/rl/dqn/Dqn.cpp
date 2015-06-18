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
    m_target_net_update_freq = 10000;
    m_replay_memory = replay_memory(m_replay_size);

    m_epsilon_beginning = 1.0;
    m_epsilon_end = 0.1;
    m_end_exploration = 1000000;

    m_picked = std::vector<bool>(m_replay_size);
    
    caffe::Caffe::SetDevice(0);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::DeviceQuery();
    
    // Initialize net and solver
    caffe::NetParameter *net_param = new caffe::NetParameter();
    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie("dqn_solver.prototxt", &solver_param);
    cout<<"has "<<solver_param.has_solver_type()<<endl;
    cout<<solver_param.has_lr_policy()<<endl;
    cout<<solver_param.SolverType_Name(solver_param.solver_type())<<endl;
    caffe::ReadNetParamsFromTextFileOrDie("dqn_net.prototxt",net_param);
    cout<<"Setting the correct number of actions..."<<endl;
    for(int i = 0;i<net_param->layers_size();i++){
        auto lay = net_param->mutable_layers(i);
        if(lay->top_size()>0&&lay->top(0)=="q_values"){
            auto inner = lay->mutable_inner_product_param();
            inner->set_num_output(numActions);
        }
    }
    solver_param.set_allocated_net_param(net_param);
    cout<<"Configuring first solver"<<endl;
    m_solver.reset(caffe::GetSolver<float>(solver_param));
    cout<<"Configuring second solver"<<endl;
    m_solver_hat.reset(caffe::GetSolver<float>(solver_param));
    m_net = m_solver->net();
    m_net_hat = m_solver_hat->net();
    m_frame_input_layer =  boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>( m_net->layer_by_name("frames_input_layer"));
    m_frame_input_layer_hat =  boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>( m_net_hat->layer_by_name("frames_input_layer"));
    m_target_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(m_net->layer_by_name("target_input_layer"));
    m_target_input_layer_hat = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(m_net_hat->layer_by_name("target_input_layer"));
    m_q_values_blob = m_net->blob_by_name("q_values");
    m_q_values_blob_hat = m_net_hat->blob_by_name("q_values");
    m_Q = vector<double>(numActions);

    m_target_buff = new float[m_batchSize*numActions];
    m_target_buff_hat = new float[m_batchSize*numActions];

}
DqnLearner::~DqnLearner(){
    delete[] m_target_buff;
    delete[] m_target_buff_hat;
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
    this->epsilon = m_epsilon_beginning;
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

    int nb_frames_played = 0;
	for(int episode = 0; totalNumberFrames < totalNumberOfFramesToLearn; episode++){
		//Repeat(for each step of episode) until game is over:
        int currentAction = 0;
        gettimeofday(&tvBegin, NULL);
        int step = -1;
        reward = 0;
		while(!env.isTerminal()){
            step++;
            //check if we have to update target network
            if(nb_frames_played >0 && nb_frames_played%m_target_net_update_freq == 0){
                updateTargetNet();
            }
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
            //at this point, reward contain the accumulated reward over the past skipped frames. We have to clip it, store it (crediting the last action taken), and reinit it.
            reward = min(1.0,reward);
            reward = max(-1.0,reward);
            m_replay_memory.storeReward(reward);
            reward = 0;
            //acknowledge that the saving is successfull
            correctlySaved = true;
            
            //store the current frame in the buffer
            std::swap(current_frame,frame_buffer[frame_buffer_index]);
            frame_buffer_size = (frame_buffer_size+1 < m_numFramesPerInput) ? frame_buffer_index+1 : m_numFramesPerInput;
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
            //anneal epsilon
            this->epsilon = m_epsilon_end + max(0.0,(m_epsilon_beginning-m_epsilon_end)*(1.0-double(nb_frames_played)/double(m_end_exploration)));
            currentAction = epsilonGreedy(m_Q);
            
            correctlySaved = false;
            firstActionTaken = true;
            //cout<<"playing with "<<currentAction<<endl;
			//Take action, observe reward and next state:
			reward += env.act(actions[currentAction]);
            nb_frames_played++;
            miniBatchLearning();            
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
            frame_buffer_size = (frame_buffer_size+1 < m_numFramesPerInput) ? frame_buffer_index+1 : m_numFramesPerInput;
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


void DqnLearner::updateTargetNet()
{
    cout<<"restoring"<<endl;
    m_solver->Snapshot("weight_snap.wg");
    m_solver_hat->Restore("weight_snap.wg.solverstate");
}

void DqnLearner::miniBatchFeed(int t, int pos_in_batch)
{
    //we start by decompressing all the relevant frames
    miniBatchUncompressFrames(t);
    const int frame_offset = m_imageDim*m_imageDim;
    const int batch_offset = frame_offset*m_numFramesPerInput;

    //copy the data in the input blob of the policy net
    std::copy(decompress_buff,decompress_buff+(frame_offset*m_numFramesPerInput),m_input_buff + pos_in_batch*batch_offset);

    //if t is not terminal, we also copy data to the input blob of the target net
    std::copy(decompress_buff+frame_offset,decompress_buff+(frame_offset*(m_numFramesPerInput+1)),m_input_buff_hat + pos_in_batch*batch_offset);

    
}

void DqnLearner::miniBatchUncompressFrames(int t)
{
    //This buffer is meant to contain all the decompressed frames that we need.
    //We need the history of the state t, which is the m_numframesperinput-1 frames before it, and the frame at t itself
    //We also need the history of state t+1, but it overlaps with the one of t : we only need the frame t+1
    //In all, there are m_numframesperinput+1 frames to decompress, and we will store them chronologically
    memset(decompress_buff,0,m_imageDim*m_imageDim*(m_numFramesPerInput+1)*sizeof(char));
    const int offset=m_imageDim*m_imageDim;
    if(!m_replay_memory.terminations[t]){
        //if t is not terminal, we also decompress frame t+1 in the last slot.
        int indice = (t+1) % m_replay_size;
        snappy::RawUncompress(m_replay_memory.frames[indice].data(),m_replay_memory.frames[indice].size(),reinterpret_cast<char*>(decompress_buff+offset*(m_numFramesPerInput)));
    }
    //Then we decompress history, starting at t
    for(int i = 0;i<m_numFramesPerInput;i++){
        int indice = t-i;
        if(indice<0){
            if(m_replay_memory.num_stored!=m_replay_size){
                //we reached the beginning of history, there is nothing before that.
                break;
            }
            indice +=  m_replay_size;
        }
        if(t>=m_replay_memory.cur_pos && indice<m_replay_memory.cur_pos){
            //in this case, the replay memory is full, and we reached the oldest frame in it: we can't go further
            break;
        }
        snappy::RawUncompress(m_replay_memory.frames[indice].data(),m_replay_memory.frames[indice].size(),reinterpret_cast<char*>(decompress_buff+offset*(m_numFramesPerInput-i-1)));
        
        
    }
}

void DqnLearner::miniBatchLearning()
{
    cout<<m_replay_memory.num_stored<<endl;
    if(m_replay_memory.num_stored-1>m_batchSize){
        cout<<"minibatch"<<endl;
        //reset the input blobs
        memset(m_input_buff,0,m_numFramesPerInput*m_imageDim*m_imageDim*m_batchSize*sizeof(float));
        memset(m_input_buff_hat,0,m_numFramesPerInput*m_imageDim*m_imageDim*m_batchSize*sizeof(float));
        memset(m_target_buff,-1, numActions*m_batchSize*sizeof(float));
            
        std::fill(m_picked.begin(), m_picked.end(), false);
        vector<int> pool(m_batchSize); //list of the samples
        int pos = 0;
        while(pos<m_batchSize){
            int candidate;
            if(m_replay_memory.num_stored == m_replay_size){ //mem is full, we can pick from anywhere
                candidate = rand()% m_replay_size;
            }else{
                candidate = rand()%(m_replay_memory.cur_pos-1); //we can't pick the last frame, because we need to know the frame that comes after
            }
            if(!m_picked[candidate]){
                m_picked[candidate]=true;
                pool[pos] = candidate;
                miniBatchFeed(candidate, pos);
                pos++;
            }
        }
        //we run one forward computation on the behavior net to obtain the current Q values
        m_frame_input_layer->Reset(m_input_buff,m_dummy_labels,m_batchSize);
        m_target_input_layer->Reset(m_target_buff,m_dummy_labels,m_batchSize);
        m_net->ForwardPrefilled();
        //we retrieve the data
        const float *q_from_net = m_q_values_blob->cpu_data();
        for(int i=0;i<m_batchSize*numActions; i++){
            cout<<q_from_net[i]<<" ";
        }
        cout<<endl<<endl<<endl;
        //most of the current q values are going to be unchanged, we let them be the target values
        const int offset = numActions;
        std::copy(q_from_net,q_from_net+(m_batchSize*offset),m_target_buff);

        //we run a forward computation in the target net 
        m_frame_input_layer_hat->Reset(m_input_buff_hat, m_dummy_labels_hat, m_batchSize);
        m_target_input_layer_hat->Reset(m_target_buff_hat, m_dummy_labels_hat, m_batchSize);
        m_net_hat->ForwardPrefilled();
        const float *q_from_net_hat = m_q_values_blob_hat->cpu_data();

        //we compute the targets for the actions that were actually taken
        for(int i=0;i<m_batchSize;i++){
            int action_taken = m_replay_memory.actions[pool[i]];
            float target = m_replay_memory.rewards[pool[i]];
            if(!m_replay_memory.terminations[pool[i]]){
                //we find the best value in target q function
                float best = -1e9;
                for(int j=0; j<numActions; j++){
                    best = max(best,q_from_net_hat[ (i*numActions)+j ]);
                }
                target += best; 
            }
            //update the corresponding target
            m_target_buff[(i*numActions) + action_taken] = target;
        }

        //we update the target blob
        m_frame_input_layer->Reset(m_input_buff,m_dummy_labels,m_batchSize);
        m_target_input_layer->Reset(m_target_buff,m_dummy_labels,m_batchSize);
        for(int i=0;i<m_batchSize*numActions; i++){
            cout<<m_target_buff[i]<<" ";
        }

        //we do one step of computation
        m_solver->Step(1);

    }
}
