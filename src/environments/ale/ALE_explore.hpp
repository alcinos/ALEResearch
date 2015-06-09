/**
 * @file   ALE_explore.hpp
 * @author Nicolas Carion
 * @date   Tue Jun  9 11:53:18 2015
 *
 * @brief  This class describe a modified ale environment designed to favor exploration
 *
 *
 */
#ifndef ALEEXPL_H
#define ALEEXPL_H

#include "ALEEnvironment.hpp"
#include<map>
#include<vector>
#include "../../common/Ntsc.hpp"
#include "../../common/Parameters.hpp"

template < typename FeatureComputer>
class ALEExplore;

static constexpr std::array<uint8_t,256> m_ntsc_to_grayscale = NtscConversionTableGenerator();;

template < typename FeatureComputer>
class ALEExplore : public ALEEnvironment<FeatureComputer>
{
public:
    ALEExplore(ALEInterface* ale,FeatureComputer* feat, Parameters* param) : ALEEnvironment<FeatureComputer>(ale,feat){back = new Background(param);}

    virtual double act(Action action){
        this->m_ale->getLegalActionSet();
        char v = compute_stats();
        return max(0.0,1.0 - double(v)/10.0);
    }
protected:
    char compute_stats(){

        const auto old_height =  this->m_ale->getScreen().height();
        const auto old_width = this->m_ale->getScreen().width();
        unsigned char* rawScreen = this->m_ale->getScreen().getArray();
        for(unsigned int i = 0; i < old_height; i++){
            for(unsigned int j = 0; j < old_width; j++){
                if((rawScreen[i*old_width+j]>>1) == (back->getPixel(i,j)>>1)){
                    rawScreen[i*old_width+j] = 0;
                }else{
                    rawScreen[i*old_width+j] = m_ntsc_to_grayscale[ rawScreen[i*old_width+j]];
                }
            }
        }

        //we now have to make a scaling.
        //we use bilinear scaling
        const auto new_height = 84;
        const auto new_width = 84;
        vector<unsigned char> feat(new_height*new_width);

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
        vector<bool> img(feat.size());
        transform(feat.begin(),feat.end(),img.begin(),[](unsigned char a){return a>128;});
        return ++visit_stats[img];
        
    }
    

    Background* back;
    std::map<std::vector<bool>,char> visit_stats;
};

#endif
