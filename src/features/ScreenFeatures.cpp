/**
 * @file   ScreenFeatures.cpp
 * @author Nicolas Carion
 * @date   Sat Jun  6 22:03:19 2015
 *
 * @brief  Implementation file of the Screen Features of the ale.
 *
 *
 */


#include "ScreenFeatures.hpp"

using namespace std;
constexpr std::array<uint8_t,256> ScreenFeatures::m_ntsc_to_grayscale;

ScreenFeatures::ScreenFeatures(Parameters* param) :
    previousScreen(NULL)
{
    this->param = param;

}

ScreenFeatures::~ScreenFeatures(){
    if(previousScreen){
        delete[] previousScreen;
    }
}

void ScreenFeatures::getRawFeatures(vector<uint8_t> & feat, ALEInterface* ale){
    unsigned char* rawScreen = ale->getScreen().getArray();
    const auto old_height =  ale->getScreen().height();
    const auto old_width = ale->getScreen().width();
    const auto old_size = old_height*old_width;
    if(step % param->getNumStepsPerAction() == 0){
        if(!previousScreen){
            previousScreen = new unsigned char[old_size];
            memset(previousScreen,0,old_size*sizeof(unsigned char));
        }
        //take max with previous frame and convert to grayscale
        for(unsigned i = 0; i< old_size;i++){
            rawScreen[i] = m_ntsc_to_grayscale[max(rawScreen[i],previousScreen[i])];
        }

        //we now have to make a scaling.
        //we use bilinear scaling
        const auto new_height = 84;
        const auto new_width = 84;
        feat.clear();
        feat.resize(new_height*new_width);

        float x_ratio = ((float)(old_width-1))/new_width ;
        float y_ratio = ((float)(old_height-1))/new_height ;
        int offset = 0 ;
        for (int i=0;i<new_height;i++) {
            for (int j=0;j<new_width;j++) {
                int x = (int)(x_ratio * j) ;
                int y = (int)(y_ratio * i) ;
                float x_diff = (x_ratio * j) - x ;
                float y_diff = (y_ratio * i) - y ;
                int index = y*old_width+x ;

                auto A = rawScreen[index];
                auto B = rawScreen[index+1];
                auto C = rawScreen[index+old_width];
                auto D = rawScreen[index+old_width+1];

                // Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
                uint8_t gray = (uint8_t)(
                                         A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                                         C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                                         ) ;

                feat[offset++] = gray ;
            }
        }
    }
    int fskip = param->getNumStepsPerAction();
    if(fskip == 1 || (step%fskip) == (fskip-1))
        memcpy(previousScreen,rawScreen,old_size*sizeof(unsigned char));

    step++;
}
