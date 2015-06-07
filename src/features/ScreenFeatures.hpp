/**
 * @file   ScreenFeatures.hpp
 * @author Nicolas Carion
 * @date   Sat Jun  6 21:38:02 2015
 *
 * @brief  This is the implementation of a proxy to the raw screen of the Atari, meant to be used in the dqn setting.
 *
 *
 */


#ifndef SCREENFEATURES_H
#define SCREENFEATURES_H
#include "Features.hpp"
#include "Background.hpp"
#include "../common/Ntsc.hpp"
#include <array>
class ScreenFeatures : public Features::Features{
private:
    Parameters *param;
    int numberOfFeatures;

    static constexpr std::array<uint8_t,256> m_ntsc_to_grayscale = NtscConversionTableGenerator();
public:
    typedef uint8_t FeatureType;
    /**
     * Destructor, used to delete the background, which is allocated dynamically.
     */
    ~ScreenFeatures();
    /**
     * Constructor. Since every operation in this class has to be done knowing the number of
     * columns, rows and colors to generate the feature vector, this information is given in
     * param, as any other relevant information such as background.
     *
     * @param Parameters *param, which gives access to the number of columns, number of rows,
     *                   number of colors and the background information
     * @return nothing, it is a constructor.
     */
    ScreenFeatures(Parameters *param);
    void getActiveFeaturesIndices(const ALEScreen &screen, const ALERAM &ram, vector<int>& features){};
    /**
     * Return the number of features returned. In this case, it is the number of pixels, 84*84
     * @param none.
     * @return int number of features generated by this method.
     */
    int getNumberOfFeatures(){return numberOfFeatures;}

    /** fill the features vector, using the ale.
     *
     *
     * @param features return parameter where the features are to be stored
     * @param ale pointer to the interface of the arcade learning environment
     */
    void getRawFeatures(vector<FeatureType>& features,ALEInterface* ale);
};


#endif
