/****************************************************************************************
** Implementation of RAM Features, described in details in the paper below. 
**       "The Arcade Learning Environment: An Evaluation Platform for General Agents.
**        Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.
**        Journal of Artificial Intelligence Research, 47:253–279, 2013."
**
** The idea is to get the RAM state and each bit be a feature in the feature vector.
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef RAMFEATURES_H
#define RAMFEATURES_H
#include "Features.hpp"

class RAMFeatures : public Features::Features{
	private:
	public:
    typedef bool FeatureType;
		/**
 		* Constructor. 
 		*
 		* @return nothing, it is a constructor.
 		*/
		RAMFeatures();
		/**
 		* This method is the instantiation of the virtual method in the class Features (also check
 		* its documentation). It gets a the current RAM state and stores the indices that correspond
 		* to active bits in the RAM.
 		*
 		* REMARKS: - It is necessary to provide both the screen and the ram because of the superclass,
 		* despite the Screen being useless here. In fact a null pointer works just fine.
 		*          - To avoid return huge vectors, this method is void and the appropriate
 		* vector is returned trough a parameter passed by reference.
 		* 
 		* @param ALEScreen &screen is the current game screen that one may use to extract features.
 		* @param ALERAM &ram is the current game RAM that one may use to extract features.
 		* @param vector<int>& features an empy vector that will be filled with the requested information,
 		*        therefore it must be passed by reference. Its i-th position is TRUE if the i-th feature is active.
 		* @return nothing as one will receive the requested data by the last parameter, by reference.
 		*/
		void getActiveFeaturesIndices(const ALEScreen &screen, const ALERAM &ram, std::vector<int>& features);	
		/**
 		* Obtain the total number of features that are generated by this feature representation.
 		*
 		* @param none.
 		* @return int number of features generated by this method.
 		*/
		int getNumberOfFeatures();
		/**
		* Destructor, not necessary in this class.
		*/
		~RAMFeatures();
};


#endif
