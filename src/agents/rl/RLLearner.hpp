/****************************************************************************************
** Abstract class that needs to be implemented by any ALE agent. It defines the method 
** that controls the agent, forcing it to act. By using it we ensure that any approach,
** based in RL can be easily run, following the same pattern.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef RLLEARNER_H
#define RLLEARNER_H
#include "../Agent.hpp"
#include "../../environments/Environment.hpp"
#include "../../common/Mathematics.hpp"


template<typename FeatureType>
class RLLearner;

template<typename FeatureType>
class RLLearner : public Agent<FeatureType>{
	protected:
		ActionVect actions;

		double gamma, epsilon;
		double firstReward;
		bool   sawFirstReward;

		int    toUseOnlyRewardSign, toBeOptimistic;
		int    randomActionTaken, numActions;
		int    episodeLength, numEpisodesEval;
		int    totalNumberOfFramesToLearn;

		/**
 		* It acts in the environment and makes the proper operations in the reward signal (normalizing,
 		* being optimistic, etc).
 		*
 		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		* @param int action action to be taken
 		*
 		* @param vector<double>& reward this vector is used to return, by reference, the reward observed
 		* by executing the action defined as parameter. This vector has two positions: in the first position
 		* the reward to be used by the RL algorithm is returned; in the second position, the game score is
 		* returned.
 		*/
    void act(Environment<FeatureType>& env, int action, vector<double> &reward);

		/**
 		* Implementation of an epsilon-greedy function. Epsilon is defined in the constructor,
 		* in the argument Parameters *param
 		*
 		* @return int action to be taken
 		*/
		int epsilonGreedy(vector<double> &QValues);

		/**
		* Constructor to be used by the RL classes to save the parameters that
		* will be used by other methods.
		*
		* @param ALEInterface& ale Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		* @param Parameters *param object containing the parameters passed to the algorithm, both by
 		*        file and command line.
 		*
		*/
    RLLearner(Environment<FeatureType>& env, Parameters *param,RLLearner<FeatureType>* off_policy=NULL);


    RLLearner<FeatureType>* offlineLearner; //in case an other agent is learning offline from another's actions

public:
    /**This function is used in the scenario where an agent is playing, but we trained off-policy another at the same time. 
     * It is called when a step is taken by the active agent
     * Note that by default, an agent is assumed not to provide such a way of being trained.
     * @param active_features_before description of the state before the action
     * @param action_taken action taken by the agent 
     * @param reward reward obtained
     * @param active_feature_after description of the state after the action
     */
    virtual void stepTaken(std::vector<int>& active_features_before,int action_taken,float reward,std::vector<int>& active_feature_after){
        throw std::runtime_error("this method doesn't support off-line learning");
    }
    /**This function is used in the scenario where an agent is playing, but we trained off-policy another at the same time
     *It is called when a new episode starts (it means implicitly that the previous one has ended).
     */
    virtual void beginEpisode(){
        throw std::runtime_error("this method doesn't support off-line learning");
    }

	   /**
 		* Pure virtual method that needs to be implemented by any RL agent.
 		*
 		* TODO: it may be useful to return something for the caller, as the total reward or policy. 
 		*       Additionally, it may be important to persist what was learned, maybe with another
 		*       pure virtual method?
 		*
 		* @param ALEInterface& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator's screen, RAM, etc.
 		*/
    virtual void learnPolicy(Environment<FeatureType>& env) = 0;

		/**
 		* Pure virtual method that needs to be implemented by any agent. Once the agent learned a
 		* policy it executes this policy for a given number of episodes. The policy is stored in
 		* the class' object.
 		*
 		* TODO it may be useful to return something for the caller, as an indicator of performance. 
 		*
 		* @param ALEInterface& env Arcade Learning Environment interface: object used to define agents'
 		*        actions, obtain simulator screen, RAM, etc.
 		*/
    virtual void evaluatePolicy(Environment<FeatureType>& env) = 0;

		/**
		* Destructor, not necessary in this class.
		*/
		virtual ~RLLearner(){};
};


#include "RLLearner.cpp"
#endif
