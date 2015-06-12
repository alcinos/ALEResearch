/**
 * @file   Dqn.hpp
 * @author Nicolas Carion
 * @date   Fri Jun  5 16:30:09 2015
 * 
 * @brief  This is an implementation of a Deep Q learning agent, initially based on DeepMind's Nature paper.
 * 
 * 
 */

#ifndef DQN_H
#define DQN_H

#include "../RLLearner.hpp"
#include <vector>
#include <caffe/caffe.hpp>

using Pixel = uint8_t;



class DqnLearner : public RLLearner<Pixel>
{
public:
    //this constructor is made unavailable because the user has to provide the parameters and the environment
    DqnLearner() = delete;
    
    DqnLearner(Environment<Pixel>& env, Parameters *param);

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
    std::shared_ptr<caffe::Solver<float>> m_solver;
    boost::shared_ptr<caffe::Net<float>> m_net;

    boost::shared_ptr<caffe::MemoryDataLayer<float>> m_frame_input_layer;
    boost::shared_ptr<caffe::MemoryDataLayer<float>> m_target_input_layer;
    boost::shared_ptr<caffe::Blob<float>> m_q_values_blob;

    std::vector<double> m_Q;
    double alpha,lambda;
    int m_playFreq;
    static constexpr int m_numFramesPerInput = 4;// this is the number of stacked frames given to the network as input
    static constexpr int m_imageDim = 84;
    static constexpr int m_batchSize = 32;
    static constexpr int m_numActions = 18;

    //buffer in wich we are going to write the data to be fed to the network
    float m_input_buff[m_numFramesPerInput*m_imageDim*m_imageDim*m_batchSize];

    //caffe needs labels as input, we provide it a pointer to dummy data.
    float m_dummy_labels[m_batchSize];

    //buffer containing the target value (to compute the loss)
    float m_target_buff[m_batchSize*m_numActions];
    
    void feedNet(std::array<std::vector<Pixel>, m_numFramesPerInput>& buffer, int current_buffer_index);

    void updateQValues();
};


#endif
