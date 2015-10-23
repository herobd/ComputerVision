#include "customsift.h"

CustomSIFT* CustomSIFT::singleton=NULL;
map<double,Mat> CustomSIFT::blurred;

//TODO this should blur the image before sampling. Maybe.
Vec2f CustomSIFT::getSubpix(const Mat& img, Point2f off,KeyPoint p)
{
    Mat patch;
    Point2f pt (off.x+p.pt.x, off.y+p.pt.y);
    getRectSubPix(img, cv::Size(1,1), pt, patch);
    return patch.at<Vec2f>(0,0);
}

float CustomSIFT::getSubpixBlur(const Mat& img, Point2f off,KeyPoint p, double blur)
{
    Mat use;
    if (blur < 1)
    {
        use=img;
    }
    else 
    {
        if (blurred.find(blur) == blurred.end())
        {
            Mat b;
            int size = (int)std::ceil(2*blur);
            size += size%2==0?1:0;
            GaussianBlur(img,b,Size(size,size),blur,blur);
            blurred[blur]=b;
        }
        use = blurred[blur];
    }
    Mat patch;
    Point2f pt (off.x+p.pt.x, off.y+p.pt.y);
    getRectSubPix(use, cv::Size(1,1), pt, patch);
    return patch.at<float>(0,0);
}

double CustomSIFT::guassianWeight(double dist, double spread)
{
    return exp(-1*dist/(2*spread*spread));
}

void CustomSIFT::normalizeDesc(vector<double>& desc)
{
    double sum=0;
    for (double v : desc)
    {
        sum+=v*v;    
    }
    double norm = sqrt(sum);
    if (norm!=0)
    for (int i=0; i<desc.size(); i++)
    {
        desc[i] /= norm;
    }
}

Point2f CustomSIFT::rotatePoint(Mat M, const Point2f& p)
{ 
    Mat_<double> src(3/*rows*/,1 /* cols */); 

    src(0,0)=p.x; 
    src(1,0)=p.y; 
    src(2,0)=1.0; 

    Mat_<double> dst = M*src; //USE MATRIX ALGEBRA 
    return Point2f(dst(0,0),dst(1,0)); 
    
} 

void CustomSIFT::extract(const Mat &img, const vector<KeyPoint> &keyPoints, vector< vector<double> >& descriptors)
{
    int descInGrid=4;
    int numBins=8;
    //Mat gradImg = computeGradient(img);
    
    descriptors.resize(keyPoints.size());
    for (int kpi=0; kpi<keyPoints.size(); kpi++)
    {
        KeyPoint p = keyPoints[kpi];
        double scale = p.size/(16.0);
        //Mat rotM = getRotationMatrix2D(p.pt,p.angle,1);
        
        double rot = CV_PI*p.angle/180;
        Mat_<double> rotM = (Mat_<double>(3, 3) << cos(rot), -sin(rot), 0, 
                                                   sin(rot),  cos(rot), 0,
                                                          0,         0, 1);
        
        descriptors[kpi].resize(descInGrid*descInGrid*numBins);
        descriptors[kpi].assign(descInGrid*descInGrid*numBins,0);
        
        
        
        for (int boxR=-descInGrid/2; boxR<descInGrid/2; boxR++)
            for (int boxC=-descInGrid/2; boxC<descInGrid/2; boxC++)
            {
                for (int xOffset=-3.5; xOffset<=3.5; xOffset+=1.0)
                {
                    for (int yOffset=-3.5; yOffset<=3.5; yOffset+=1.0)
                    { 
                        Point2f actualOffset = rotatePoint(rotM,Point2f((4*boxC+xOffset)*scale ,(4*boxR+yOffset)*scale));
                        
                        Point2f actualOffset_hp = rotatePoint(rotM,Point2f((4*boxC+xOffset+1)*scale ,(4*boxR+yOffset)*scale));
                        Point2f actualOffset_hn = rotatePoint(rotM,Point2f((4*boxC+xOffset-1)*scale ,(4*boxR+yOffset)*scale));
                        
                        Point2f actualOffset_vp = rotatePoint(rotM,Point2f((4*boxC+xOffset)*scale ,(4*boxR+yOffset+1)*scale));
                        Point2f actualOffset_vn = rotatePoint(rotM,Point2f((4*boxC+xOffset)*scale ,(4*boxR+yOffset-1)*scale));
                        
                        double firstDist = sqrt( pow((4*boxC+xOffset)*scale,2) + pow((4*boxR+yOffset)*scale,2) );
                        double actualDist = sqrt(actualOffset.x*actualOffset.x + actualOffset.y*actualOffset.y);
                        assert (fabs(firstDist-actualDist)<.001);
                        //Vec2f v = getSubpix(img,actualOffset,p);
                        double hGrad = getSubpixBlur(img,actualOffset_hp,p,scale) - getSubpixBlur(img,actualOffset_hn,p,scale);
                        double vGrad = getSubpixBlur(img,actualOffset_vp,p,scale) - getSubpixBlur(img,actualOffset_vn,p,scale);
                        double mag = sqrt(hGrad*hGrad + vGrad*vGrad) * guassianWeight(actualDist,p.size/2.0);
                        double theta = atan2(vGrad,hGrad);
                        //theta += CV_PI*p.angle/180;
                        
                        double binSpace = (theta+CV_PI)/(2*CV_PI) * numBins;
                        int binCenter = ((int)(binSpace + (0.5)))%numBins;
                        double distCenter = min(fabs(binCenter-binSpace), min(fabs((binCenter-8.0)-binSpace), fabs((binCenter+8.0)-binSpace)));
                        int binUp = (binCenter+1)%numBins;
                        double distUp = min(fabs(binUp-binSpace), min(fabs((binUp-8.0)-binSpace), fabs((binUp+8.0)-binSpace)));
                        int binDown = mod(binCenter-1,numBins);
                        double distDown = min(fabs(binDown-binSpace), min(fabs((binDown-8.0)-binSpace), fabs((binDown+8.0)-binSpace)));
                        
                        
                        descriptors[kpi].at((boxC+descInGrid/2)*descInGrid*numBins + 
                                         (boxR+descInGrid/2)*numBins + 
                                         binCenter) += (1.5-distCenter)*mag;
                        //cout << "addedC " << (1.5-distCenter)*mag << endl;
                        descriptors[kpi].at((boxC+descInGrid/2)*descInGrid*numBins + 
                                         (boxR+descInGrid/2)*numBins + 
                                         binUp) += (1.5-distUp)*mag;
                        //cout << "addedU " << (1.5-distUp)*mag << endl;
                        descriptors[kpi].at((boxC+descInGrid/2)*descInGrid*numBins + 
                                         (boxR+descInGrid/2)*numBins + 
                                         binDown) += (1.5-distDown)*mag;
                        //cout << "addedD " << (1.5-distDown)*mag << endl;
                        
                        assert(distDown > distCenter && distUp > distCenter);
                        assert(1.5-distCenter >= 0);
                        assert(1.5-distUp >= 0);
                        assert(1.5-distDown >= 0);
                    }
                    
                }
            }
        
        
        normalizeDesc(descriptors[kpi]);
        for (int i=0; i<descriptors[kpi].size(); i++)
        {
            //descriptors[kpi][i] = (descriptors[kpi][i]-min)/max;
            if (descriptors[kpi][i] > 0.2)
                descriptors[kpi][i] = 0.2;
        }
        normalizeDesc(descriptors[kpi]);
        
    }
    
    assert(descriptors.size() == keyPoints.size());
    assert(descriptors[0].size() == 128);
    //cout << "extracted " << descriptors.size() <<endl;
}

Mat CustomSIFT::computeGradient(const Mat &img)
{
    Mat h = (Mat_<double>(1,3) << -1, 0, 1);
    Mat v = (Mat_<double>(3,1) << -1, 0, 1);
    
    Mat h_grad;
    filter2D(img,h_grad,CV_32F,h);
    
    Mat v_grad;
    filter2D(img,v_grad,CV_32F,v);
//    h_grad=cv::abs(h_grad);
//    v_grad=cv::abs(v_grad);
    
    Mat chan[2] = {h_grad, v_grad};
    Mat ret;
    merge(chan,2,ret);
    return ret;
}
