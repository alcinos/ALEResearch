
#include "common/Parameters.hpp"
#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "agents/rl/sarsaSVD/SarsaSVD.hpp"
#include "environments/mountainCar/MountainCarEnvironment.hpp"
#include "features/MountainCarFeatures.hpp"

double curIter;

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

using namespace std;
int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());
	
	MountainCarFeatures features;
	//Reporting parameters read:
	printBasicInfo(param);
	
    MountainCarEnvironment<MountainCarFeatures> env(&features);
    float scores[5];
    param.setLearningLength(200);
    for(int i=0;i<5;i++){
        cout<<endl<<"FLAVOR "<<i<<endl;
        env.setFlavor(i);
        //Instantiating the learning algorithm:
        SarsaLearner sarsaLearner(env,&param);
        //Learn a policy:
        sarsaLearner.learnPolicy(env);
        //sarsaLearner.showGreedyPol();
        printf("\n\n== Evaluation without Learning == \n\n");
        scores[i] = sarsaLearner.evaluatePolicy(env,50);
    }
    cout<<"Final Scores "<<endl;
    param.setLearningLength(1000);
    for(int i=0;i<5;i++){
        cout<<scores[i]<<endl;
        }
    
    SarsaSVD sarsaSVD(env,&param,5);
    sarsaSVD.learnPolicy(env);
    sarsaSVD.evaluatePolicy(env);
    
    return 0;
}