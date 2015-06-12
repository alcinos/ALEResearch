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

void DqnLearner::learnPolicy(Environment<Pixel>& env){


}

void DqnLearner::feedNet(std::array<std::vector<Pixel>, m_numFramesPerInput>& input_buffer, int current_buffer_index)
{
    memset(input_buff,0,m_numFramesPerInput*m_imageDim*m_imageDim*m_batchSize*sizeof(float));
    int size_written = 0;
    //copy the data in the input blob
    for(int i = 0; i< m_numFramesPerInput; i++){
        int cur_pos = (i+current_buffer_index) % m_numFramesPerInput;
        std::copy(input_buffer[cur_pos].begin(),input_buffer[cur_pos].end(),input_buff + size_written);
        size_written += input_buffer[cur_pos].size();
    }
    m_frame_input_layer->Reset(input_buff,dummy_labels,m_batchSize);
    m_target_input_layer->Reset(target_buff,dummy_labels,m_batchSize);
    //run forward computation
    m_net->ForwardPrefilled();
}

void DqnLearner::updateQValues()
{
    const float *q_from_net = m_q_values_blob->gpu_data();
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
		for(int step = 0; !env.isTerminal() && step < episodeLength; step++){
            //get the current frame and store it in the buffer
            env.getRawFeatures(current_frame);
            std::swap(current_frame,frame_buffer[frame_buffer_index]);
            frame_buffer_size = min(frame_buffer_size+1,m_numFramesPerInput);
            frame_buffer_index = (frame_buffer_index+1) % m_numFramesPerInput;

            if(frame_buffer_size < m_numFramesPerInput){
                //we do not have enough frames to feed the net, we just noop
                env.act(PLAYER_A_NOOP);
                cout<<"playing a noop"<<endl;
                continue;
            }
            if(step % m_playFreq==0){
                feedNet(frame_buffer,frame_buffer_index);
                currentAction = epsilonGreedy(m_Q);
            }
			//Take action, observe reward and next state:
			reward = env.act(actions[currentAction]);
            //cout<<"playing"<<currentAction<<endl;
			cumReward  += reward;
		}
		gettimeofday(&tvEnd, NULL);
		timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
		elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
		double fps = double(env.getEpisodeFrameNumber())/elapsedTime;

		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n", 
			episode + 1, (cumReward-prevCumReward), (double)cumReward/(episode + 1.0), env.getEpisodeFrameNumber(), fps);

		env.reset();
		prevCumReward = cumReward;
	}
    

}


