/****************************************************************************************
** Starting point for running Sarsa algorithm. Here the parameters are set, the algorithm
** is started, as well as the features used. In fact, in order to create a new learning
** algorithm, once its class is implementend, the main file just need to instantiate
** Parameters, the Learner and the type of Features to be used. This file is a good 
** example of how to do it. A parameters file example can be seen in ../conf/sarsa.cfg.
** This is an example for other people to use: Sarsa with Basic Features.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#include <ale_interface.hpp>
#include "common/Parameters.hpp"
#include "agents/rl/qlearning/QLearner.hpp"
#include "agents/rl/dqn/Dqn.hpp"
#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "agents/rl/true_online_sarsa/TrueOnlineSarsaLearner.hpp"
#include "agents/baseline/ConstantAgent.hpp"
#include "agents/baseline/PerturbAgent.hpp"
#include "agents/baseline/RandomAgent.hpp"
#include "agents/human/HumanAgent.hpp"
#include "features/BasicFeatures.hpp"
#include "features/ScreenFeatures.hpp"
#include "environments/ale/ALEEnvironment.hpp"
#include<fstream>

void printBasicInfo(Parameters param){
	printf("Seed: %d\n", param.getSeed());
	printf("\nCommand Line Arguments:\nPath to Config. File: %s\nPath to ROM File: %s\nPath to Backg. File: %s\n", 
		param.getConfigPath().c_str(), param.getRomPath().c_str(), param.getPathToBackground().c_str());
	if(param.getSubtractBackground()){
		printf("\nBackground will be subtracted...\n");
	}
	printf("\nParameters read from Configuration File:\n");
	printf("alpha:   %f\ngamma:   %f\nepsilon: %f\nlambda:  %f\nep. length: %d\n\n", 
		param.getAlpha(), param.getGamma(), param.getEpsilon(), param.getLambda(), 
		param.getEpisodeLength());
}


int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());
	
	//Using Basic features:
	BasicFeatures features(&param);
	//Reporting parameters read:
	printBasicInfo(param);
	
	ALEInterface ale(param.getDisplay());

	ale.setFloat("stochasticity", 0.00);
	ale.setInt("random_seed", param.getSeed());
	ale.setFloat("frame_skip", param.getNumStepsPerAction());
	ale.setInt("max_num_frames_per_episode", param.getEpisodeLength());

	ale.loadROM(param.getRomPath().c_str());

    ale.act(PLAYER_A_NOOP);
    ale.act(PLAYER_A_NOOP);
    ale.act(PLAYER_A_NOOP);
    ale.act(PLAYER_A_NOOP);
    
    ScreenFeatures ff(&param);
    vector<uint8_t> p;
    ff.getRawFeatures(p,&ale);
    ofstream file("test.pgm");
    file<<"P2"<<endl;
    file<<"84 84 255"<<endl;
    for(const auto& t : p){
        file<<(int)t<<endl;
    }

    file.close();
    
    ALEEnvironment<ScreenFeatures> env(&ale,&ff);

	//Instantiating the learning algorithm:
	DqnLearner sarsaLearner(env,&param);
    //Learn a policy:
    //sarsaLearner.learnPolicy(env);

    printf("\n\n== Evaluation without Learning == \n\n");
    sarsaLearner.evaluatePolicy(env);
	
    return 0;
}
