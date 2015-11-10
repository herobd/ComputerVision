#ifndef HOG_H
#define HOG_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
//#include "defines.h"

using namespace cv;
using namespace std;

class HOG
{
public:
    HOG(float thresh, int cellSize, int stepSize=5, int num_bins=9);
    void compute(const Mat &img, vector<vector<double> > &descriptors, vector< KeyPoint > &locations);
    void unittest();
    
private:
    float thresh;
    int cellSize;
    int stepSize;
    int num_bins;
    
    Mat computeGradient(const Mat &img);
    inline int mod(int a, int b)
    {
	    while (a<0) a+=b;
	    return a%b;
    }    
};

#endif // HOG_H
