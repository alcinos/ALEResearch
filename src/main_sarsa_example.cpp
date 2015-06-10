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
#include "environments/ale/ALE_explore.hpp"
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

int true_reward;
int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());
	
	//Using Basic features:
	BasicFeatures features(&param);
	//Reporting parameters read:
	printBasicInfo(param);
	
	ALEInterface *ale = new ALEInterface(param.getDisplay());

	ale->setFloat("stochasticity", 0.00);
	ale->setInt("random_seed", param.getSeed());
	//ale->setFloat("frame_skip", param.getNumStepsPerAction());
    ale->setFloat("frame skip", 0);
    //ale->setBool("display_screen",true);
	ale->setInt("max_num_frames_per_episode", param.getEpisodeLength());

	ale->loadROM(param.getRomPath().c_str());

    ale->act(PLAYER_A_NOOP);
    ale->act(PLAYER_A_NOOP);
    ale->act(PLAYER_A_NOOP);
    ale->act(PLAYER_A_NOOP);

    const ALEScreen &screen = ale->getScreen();
    vector<vector<int> > img;
    ofstream file2("test3.pgm");
    file2<<"P2"<<endl;
    file2<<screen.width()<<" "<<screen.height()<<" 255"<<endl;
    Background* background = new Background(&param);
    for(unsigned int i = 0; i < screen.height(); i++){
        img.push_back(vector<int>(screen.width(), 0));
        for(unsigned int j = 0; j < screen.width(); j++){
            img[i][j] = screen.get(i,j);
            if((img[i][j]>>1) == (background->getPixel(i,j)>>1)){
                file2<<0<<endl;
            }else{
                file2<<img[i][j]<<endl;
            }
        }
    }
    file2.close();
    
    
    ScreenFeatures ff(&param);
    vector<uint8_t> p;
    ff.getRawFeatures(p,ale);
    ofstream file("test.pgm");
    file<<"P2"<<endl;
    file<<"84 84 255"<<endl;
    for(const auto& t : p){
        file<<(int)t<<endl;
    }

    file.close();
    
    //ALEEnvironment<ScreenFeatures> env(ale,&ff);

    ALEExplore<BasicFeatures> env(ale,&features,&param);

    QLearner qLearner(env,&param);
	//Instantiating the learning algorithm:
	SarsaLearner sarsaLearner(env,&param,&qLearner);
    //Learn a policy:
    sarsaLearner.learnPolicy(env);


    ALEInterface *ale2 = new ALEInterface(param.getDisplay());

	ale2->setFloat("stochasticity", 0.00);
	ale2->setInt("random_seed", param.getSeed());
	//ale2->setFloat("frame_skip", param.getNumStepsPerAction());
    ale2->setFloat("frame skip", 0);
    ale2->setBool("display_screen",true);
	ale2->setInt("max_num_frames_per_episode", param.getEpisodeLength());

	ale2->loadROM(param.getRomPath().c_str());
    
    ale2->setBool("display_screen",true);
    ALEEnvironment<BasicFeatures> env2(ale2,&features);
    printf("\n\n== Evaluation without Learning == \n\n");
    qLearner.evaluatePolicy(env2);
	
    return 0;
}
