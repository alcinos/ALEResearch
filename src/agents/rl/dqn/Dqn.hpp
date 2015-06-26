/**
 * @file   Dqn.hpp
 * @author Nicolas Carion
 * @date   Fri Jun  5 16:30:09 2015
 * 
 * @brief  This is an implementation of a Deep Q learning agent, initially based on DeepMind's Nature paper.
 * 
 * Implementation note about the frame skip : the frame skip is dealt with manually from within the agent. Please set it to 1 inside the ALE parameters.
 * When the frame skip is set to 1, nothing special occurs, the agent plays at each frames
 * When the frame skip is > 1, then the agent doesn't play at each frames. However, we cannot completly ignore
 * the skipped frames, because we when the agent plays, its input is a max taken over the current frame and 
 * the frame just before (to avoid flickering). Hence, we have to keep track of this penultimate frame. This is
 * handled by the feature class, but we must make sure that we call it on every frames, even the skipped ones.
 */

#ifndef DQN_H
#define DQN_H

#include "../RLLearner.hpp"
#include <vector>
#include <caffe/caffe.hpp>
#include <snappy/snappy.h>

using Pixel = uint8_t;



class DqnLearner : public RLLearner<Pixel>
{
public:
    //this constructor is made unavailable because the user has to provide the parameters and the environment
    DqnLearner() = delete;
    
    DqnLearner(Environment<Pixel>& env, Parameters *param);
    ~DqnLearner();

    /** This function is the learning function of the agent 
     * In this phase, several episodes are simulated, and the weight of the deep neural network are updated.
     * 
     * @param env a reference on the environment on which the agent is supposed to play
     */
    void learnPolicy(Environment<Pixel>& env);

    /** In this phase, the behavior of the agent is fixed with respect to the neural network (the weights are 
     * no longer updated), and we assess the performance of this agent.
     * 
     * @param env 
     */
    void evaluatePolicy(Environment<Pixel>& env);

protected:

    struct replay_memory
    {
        replay_memory(){cur_pos = 0; num_stored = 0;}
        replay_memory(int size){
            cur_pos = 0; num_stored = 0;
            frames.resize(size);
            rewards.resize(size);
            actions.resize(size);
            terminations.resize(size);
        }
        void clear(){cur_pos = 0;num_stored = 0;}
        void storeFrame(const std::vector<Pixel>& frame);
        void storeReward(float reward);
        void storeAction(int);
        void storeTermination(bool);
        int num_stored;
        std::vector<std::string> frames;
        std::vector<float> rewards;
        std::vector<int> actions;
        std::vector<bool> terminations;
        int cur_pos;
    };

    int m_replay_size;
    int m_target_net_update_freq;
    replay_memory m_replay_memory;
    double m_epsilon_beginning,m_epsilon_end;
    int m_end_exploration;
    int m_SGDFrequency;
    int m_frames_per_epoch;
    static constexpr int m_numFramesPerInput = 4;// this is the number of stacked frames given to the network as input
    static constexpr int m_imageDim = 84;
    static constexpr int m_batchSize = 32;
    
    std::shared_ptr<caffe::Solver<float>> m_solver;
    std::shared_ptr<caffe::Solver<float>> m_solver_hat;
    boost::shared_ptr<caffe::Net<float>> m_net;
    boost::shared_ptr<caffe::Net<float>> m_net_hat;

    boost::shared_ptr<caffe::MemoryDataLayer<float>> m_frame_input_layer;
    boost::shared_ptr<caffe::MemoryDataLayer<float>> m_frame_input_layer_hat;
    boost::shared_ptr<caffe::MemoryDataLayer<float>> m_target_input_layer;
    boost::shared_ptr<caffe::MemoryDataLayer<float>> m_target_input_layer_hat;
    boost::shared_ptr<caffe::Blob<float>> m_q_values_blob;
    boost::shared_ptr<caffe::Blob<float>> m_q_values_blob_hat;

    std::vector<double> m_Q;
    std::vector<bool> m_picked; //used for minibatch
    unsigned char decompress_buff[m_imageDim*m_imageDim*(m_numFramesPerInput+1)]; //used for minibatch
    double alpha,lambda;
    int m_playFreq;

    //buffer in wich we are going to write the data to be fed to the network
    float m_input_buff[m_numFramesPerInput*m_imageDim*m_imageDim*m_batchSize];
    float m_input_buff_hat[m_numFramesPerInput*m_imageDim*m_imageDim*m_batchSize];

    //caffe needs labels as input, we provide it a pointer to dummy data.
    float m_dummy_labels[m_batchSize];
    float m_dummy_labels_hat[m_batchSize];

    //buffer containing the target value (to compute the loss)
    float* m_target_buff;
    float* m_target_buff_hat;
    
    void feedNet(std::array<std::vector<Pixel>, m_numFramesPerInput>& buffer, int current_buffer_index);
    void updateTargetNet();
    void updateQValues();
    float miniBatchLearning(); //returns the loss
    void miniBatchFeed(int t, int pos_in_batch);
    void miniBatchUncompressFrames(int t);
};


#endif
