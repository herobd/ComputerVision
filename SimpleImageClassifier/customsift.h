#ifndef CUSTOMSIFT_H
#define CUSTOMSIFT_H

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

class CustomSIFT
{
public:
    static CustomSIFT* Instance()
    {
        if (singleton==NULL)
            singleton = new CustomSIFT;
        
        return singleton;
    }
    static void extract(const Mat &img, const vector<KeyPoint> &keyPoints, vector< vector<double> >& descriptors);
    
    
private:
    static CustomSIFT* singleton;
    static map<double,Mat> blurred;

    static Vec2f getSubpix(const Mat& img, Point2f off,KeyPoint p);
    static float getSubpixBlur(const Mat& img, Point2f off,KeyPoint p, double blur);
    static double guassianWeight(double dist, double spread);
    static void normalizeDesc(vector<double>& desc);
    static Mat computeGradient(const Mat &img);
    static Point2f rotatePoint(Mat M, const Point2f& p);
    
    static int mod(int a, int m) {while(a<0){a+=m;} return a%m;}
    
    CustomSIFT(){};
    CustomSIFT(CustomSIFT const&){};
    CustomSIFT& operator=(CustomSIFT const&){return *this;};
    
};

#endif // CUSTOMSIFT_H
