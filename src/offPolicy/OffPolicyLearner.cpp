#include "OffPolicyLearner.hpp"

using namespace std;

OffPolicyLearner::OffPolicyLearner(unsigned nF,const std::vector<Action>& actions, Parameters* param){
	gamma               = param->getGamma();
	epsilon             = param->getEpsilon();
    lambda = param->getLambda();
	numActions = actions.size();
    numFeatures = nF;
	traceThreshold = param->getTraceThreshold();
	alpha = param->getAlpha();
    beta = param->getBeta();
    available_actions = actions;

    episodeLength       = 180000;
	numEpisodesEval     = param->getNumEpisodesEval();
	totalNumberOfFramesToLearn = param->getLearningLength();

}


int OffPolicyLearner::epsilonGreedy(std::vector<float> &QValues){
	randomActionTaken = 0;

	int action;
	//With probability epsilon: a <- random action in A(s)
	int random = rand();
	if((random % int(nearbyint(1.0/epsilon))) == 0) {
	//if((rand()%int(1.0/epsilon)) == 0){
		randomActionTaken = 1;
		action = rand() % numActions;
	}else{
        action = Mathematics::argmax(QValues);
    }
	return action;
}

double OffPolicyLearner::evaluatePolicy(Environment<bool>& env,unsigned numSteps, bool epsilonAnneal){
	float reward = 0;
	float cumReward = 0; 
	float prevCumReward = 0;
	double elapsedTime;
    std::vector<int> F;					//Set of features active
    std::vector<int> Fnext;              //Set of features active in next state
    std::vector<float> Q(numActions);               //Q(a) entries
    std::vector<float> Qnext(numActions);           //Q(a) entries for next action

    if(epsilonAnneal)
        epsilon = 1.0;
	//Repeat (for each episode):
	for(int episode = 0; episode < numSteps; episode++){
		//Repeat(for each step of episode) until game is over:
		for(int step = 0; !env.isTerminal() && step < episodeLength; step++){
			//Get state and features active on that state:		
			F.clear();
			env.getActiveFeaturesIndices(F);
			updateQValues(F, Q);       //Update Q-values for each possible action
			int currentAction = epsilonGreedy(Q);
            //compute proba of taking current action
            //first, we need the number of QValues that are tied
            double numTies = 0;
            if(!randomActionTaken){
                for(const auto& q : Q){
                    if(q==Q[currentAction])
                        numTies++;
                }
                assert(numTies>0);
                
            }
            double proba_action = epsilon/double(numActions) + (randomActionTaken ? 0 : (1.0 - epsilon)/numTies);
			//Take action, observe reward and next state:
			reward = env.act(available_actions[currentAction],proba_action);
			cumReward  += reward;
		}

		printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t epsilon : %f\n", 
               episode + 1, (cumReward-prevCumReward), (double)cumReward/(episode + 1.0), env.getEpisodeFrameNumber(),epsilon);

		env.reset();
		prevCumReward = cumReward;
        if(epsilonAnneal)
            epsilon-=1/(double)(numSteps);
	}
    return cumReward/(double)(numSteps);
}

