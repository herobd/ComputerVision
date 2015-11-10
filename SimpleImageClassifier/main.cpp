/*Brian Daivs
 *CS601R
 *Project one: Features and Learning
 */

#include <iostream>
#include <random>
#include <tuple>
#include <regex>
#include <assert.h>
#include <dirent.h>
#include <omp.h>
#include <iomanip>
#include <cmath>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/ml/ml.hpp"

#include "svm.h"

#include "customsift.h"
#include "codebook_2.h"
#include "hog.h"

#define NORMALIZE_DESC 1

#define SAVE_LOC string("./save/")
#define DEFAULT_IMG_DIR string("./leedsbutterfly/images/")
#define DEFAULT_MASK_DIR string("./leedsbutterfly/segmentations/")
#define DEFAULT_CODEBOOK_SIZE 200

#define DEBUG_SHOW 0

using namespace std;
using namespace cv;

bool hasPrefix(const std::string& a, const std::string& pref) {
    return a.substr(0,pref.size()) == pref;
}
bool isTrainingImage(int index) {
    return index%2==1;
}
bool isTestingImage(int index) {
    return index%2==0;
}
double my_stof(string s)
{
   istringstream os(s);
   double d;
   os >> d;
   return d;
}


///////////////////////////
///DRAWING FUNCTIONS
unsigned int _seed1;
unsigned int _seed2;
void my_rand_reset()
{
    _seed1=12483589;
    _seed2=54867438;
}

unsigned int my_rand()
{
    unsigned int r = _seed1 ^ _seed2;
    _seed1 = _seed2 >> 1;
    _seed2=r;
    return r;
}
int _colorCylcleIndex=0;
Vec3b colorCylcle()
{
    //return Vec3b(255,255,255);

    _colorCylcleIndex=(_colorCylcleIndex+1)%6;
    if (_colorCylcleIndex==0)
        return Vec3b(255,0,0);
    else if (_colorCylcleIndex==1)
        return Vec3b(0,150,255);
    else if (_colorCylcleIndex==2)
        return Vec3b(0,255,0);
    else if (_colorCylcleIndex==3)
        return Vec3b(255,0,150);
    else if (_colorCylcleIndex==4)
        return Vec3b(0,0,255);
    else if (_colorCylcleIndex==5)
        return Vec3b(150,255,0);
    return Vec3b(255,255,150);
}

Mat makeHistImage(vector<double> fv)
{
    assert(NORMALIZE_DESC);
    int size=400;
    Mat hist(size*2,3*fv.size()+10,CV_8UC3);
    hist.setTo(Scalar(0,0,0));
    
    my_rand_reset();
    for (int i=0; i<fv.size(); i++)
    {
        Vec3b color((my_rand()%206)+50,(my_rand()%206)+50,(my_rand()%206)+50);
        for (double j=0; j<std::min(fabs(fv[i]*100.0),(double)size); j+=1.0)
        {
            hist.at<Vec3b>(size+std::copysign(j,fv[i]),5+3*i) = color;
            hist.at<Vec3b>(size+std::copysign(j,fv[i]),6+3*i) = color;
        }
    }
    return hist;
}

void makeHistFull(const vector<double>& fv, const vector<double>& maxs, const vector<double>& mins, Mat& full, int row)
{
    assert(NORMALIZE_DESC);
    
    _colorCylcleIndex=0;
    for (int i=0; i<fv.size(); i++)
    {
        Vec3b color=colorCylcle()*((fv[i]-mins[i])/(maxs[i]-mins[i]));
        full.at<Vec3b>(row,3*i) = color;
        full.at<Vec3b>(1+row,1+3*i) = color;
        full.at<Vec3b>(row,1+3*i) = color;
        full.at<Vec3b>(1+row,3*i) = color;
    }
}

double drawPRCurve(int label, vector<tuple<float,bool> >& data)
{
    auto comp = [](const tuple<float,bool>& a, const tuple<float,bool>& b){return get<0>(a) < get<0>(b);};
    sort(data.begin(), data.end(), comp);
    Mat curve(300,300,CV_8UC3);
    curve.setTo(Scalar(255,255,255));
    
    int countPos=0;
    for (auto t : data)
        if (get<1>(t))
            countPos++;
    
    double ap=0;
    int abovePos=0;
    Point prev;
    for (int i=0; i<data.size(); i++)
    {
        if (get<1>(data[i]))
            abovePos++;
        double precision = abovePos/(double)(i+1);
        if (get<1>(data[i]))
            ap += precision;
        double recall = abovePos/(double)(countPos);
        Point cur = Point((recall*300),300-(precision*300));
        
        if (i>0)
            line(curve,prev,cur,Scalar(0,0,150),1,CV_AA);
        circle(curve,cur,3,Scalar(0,0,150),-1,CV_AA); 
        
        prev=cur;
    }
    imwrite("pr"+to_string(label)+".png",curve);
    
    ap /=countPos;
    cout << "class "<<label<<" ap = " << ap << endl;
    return ap;
}

/////////////////////////
///REAL PROGRAM

struct svm_node* convertDescription(const vector<double>* description)
{
    int nonzeroCount=0;
    for (unsigned int j=0; j<description->size(); j++)
    {
        if (description->at(j)!=0)
            nonzeroCount++;
    }
    struct svm_node* ret = new struct svm_node[nonzeroCount+1];
    int nonzeroIter=0;
    for (unsigned int j=0; j<description->size(); j++)
    {
        if (description->at(j)!=0)
        {
            ret[nonzeroIter].index = j;
            ret[nonzeroIter].value = description->at(j);
            nonzeroIter++;
            //cout << "["<<j<<"]="<<description->at(j)<<", ";
        }
        
    }
    ret[nonzeroIter].index = -1;//end
    //cout << endl;
    return ret;
}

#define SIFT_NUMPTS 1000
#define SIFT_THRESH 0.07

vector<KeyPoint>* getKeyPoints(string option, const Mat& color_img)
{
    if (hasPrefix(option,"SIFT"))
    {
        Mat img;
        cvtColor(color_img,img,CV_BGR2GRAY);
        
        int nfeaturePoints=SIFT_NUMPTS;
        int nOctivesPerLayer=3;
        double contrastThresh=SIFT_THRESH;
        SIFT detector(nfeaturePoints,nOctivesPerLayer,contrastThresh);
        vector<KeyPoint>* ret = new vector<KeyPoint>();
        
        detector(img,noArray(),*ret,noArray(),false);
        
        //Mat out;
        //drawKeypoints( img, *ret, out, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        //imshow("o",out);
        //waitKey();
        return ret;
    }
    else if (hasPrefix(option,"dense"))
    {
        Mat img;
        cvtColor(color_img,img,CV_BGR2GRAY);
        
        int stride = 5;
        smatch sm_option;
        regex parse_option("stride=([0-9]+)");
        if(regex_search(option,sm_option,parse_option))
        {
            stride = stoi(sm_option[1]);
        }
        vector<KeyPoint>* ret = new vector<KeyPoint>();
        for (int x=stride; x<img.cols; x+=stride)
            for (int y=stride; y<img.rows; y+=stride)
                ret->push_back(KeyPoint(x,y,stride));
        return ret;
    }
    
    return NULL;
}

vector< vector<double> >* getDescriptors(string option, Mat color_img, vector<KeyPoint>* keyPoints)
{
    if (hasPrefix(option,"SIFT"))
    {
        Mat img;
        cvtColor(color_img,img,CV_BGR2GRAY);
        
        //imshow("test",img);
        
        
        int nfeaturePoints=SIFT_NUMPTS;
        int nOctivesPerLayer=3;
        double contrastThresh=SIFT_THRESH;
        SIFT detector(nfeaturePoints,nOctivesPerLayer,contrastThresh);
        Mat desc;
        detector(img,noArray(),*keyPoints,desc,true);
        vector< vector<double> >* ret = new vector< vector<double> >(desc.rows);
        assert(desc.type() == CV_32F);
	for (unsigned int i=0; i<desc.rows; i++)
        {
            ret->at(i).resize(desc.cols);
            for (unsigned int j=0; j<desc.cols; j++)
            {
                ret->at(i).at(j) = desc.at<float>(i,j);
                //cout << desc.at<float>(i,j) << ", ";
            }
            //cout << endl;
        }
        
        //waitKey();
        
        return ret;
    }
    else if (hasPrefix(option,"customSIFT"))
    {
        Mat img;
        cvtColor(color_img,img,CV_BGR2GRAY);
        
        vector< vector<double> >* ret = new vector< vector<double> >();
        CustomSIFT::Instance()->extract(img,*keyPoints,*ret);
        //cout << "custom done." << endl;
        return ret;
    }
    else if (hasPrefix(option,"HOG"))
    {
	Mat img;
	cvtColor(color_img,img,CV_BGR2GRAY);
	int stride = 5;
	int thresh = 200;
	int size = 60;
	int num_bins = 9;
	smatch sm_option;
	regex parse_option1("stride=([0-9]+)");
	if(regex_search(option,sm_option,parse_option1))
	{
	    stride = stoi(sm_option[1]);
	}
	regex parse_option2("size=([0-9]+)");
	if(regex_search(option,sm_option,parse_option2))
	{
	     size = stoi(sm_option[1]);
	}
	regex parse_option3("thresh=([0-9]+)");
	if(regex_search(option,sm_option,parse_option3))
	{
	     size = stoi(sm_option[1]);
	}
	vector< vector<double> >* ret = new vector< vector<double> >();
	if (keyPoints==NULL)
		keyPoints = new vector<KeyPoint>();
	else
		keyPoints->clear();
	HOG hog(thresh, size, stride, num_bins);
	hog.compute(img, *ret, *keyPoints);
	return ret;
    }
    
    return NULL;
}

vector<double>* getImageDescription(string option, const Mat& img, const vector<KeyPoint>* keyPoints, const vector< vector<double> >* descriptors, const Codebook* codebook)
{
    int LLC = 1;
    smatch sm_option;
    regex parse_option("LLC=([0-9]+)");
    if(regex_search(option,sm_option,parse_option))
    {
        LLC = stoi(sm_option[1]);
    }
    if (hasPrefix(option,"bovw"))
    {
        vector<double>* ret = new vector<double>(codebook->size());
        
        for (const vector<double>& desc : *descriptors)
        {
            vector< tuple<int,float> > quan = codebook->quantizeSoft(desc,LLC);
            for (const auto &v : quan)
            {
                ret->at(get<0>(v)) += get<1>(v);
            }
        }
        
        //Normalize the description
#if NORMALIZE_DESC
	    double sum;
        for (double v : *ret)
            sum += v*v;
        double norm = sqrt(sum);
        if (norm != 0)
            for (double& v : *ret)
                v /= norm;
#endif    
        return ret;
    }
    else if (hasPrefix(option,"sppy"))
    {
        vector<double>* ret = new vector<double>(codebook->size()*5);
	assert(descriptors->size() == keyPoints->size());
        for (int i=0; i<descriptors->size(); i++)
	{
            vector< tuple<int,float> > quan = codebook->quantizeSoft((descriptors->at(i)),LLC);
	    int quarter;
	    if (keyPoints->at(i).pt.x < img.cols/2 && keyPoints->at(i).pt.y < img.rows/2)
		    quarter=1;
	    else if (keyPoints->at(i).pt.x >= img.cols/2 && keyPoints->at(i).pt.y < img.rows/2)
		    quarter=2;
	    else if (keyPoints->at(i).pt.x < img.cols/2 && keyPoints->at(i).pt.y >= img.rows/2)
		    quarter=3;
	    else
		    quarter=4;
	    for (const auto &v : quan)
            {
                 ret->at(get<0>(v)) += get<1>(v);
                 ret->at(get<0>(v)+quarter*codebook->size()) += 4*get<1>(v);
            }
	}

#if NORMALIZE_DESC
	double sum;
        for (double v : *ret)
            sum += v*v;
        double norm = sqrt(sum);
        if (norm != 0)
            for (double& v : *ret)
                v /= norm;
        return ret;
#endif    
    }
    else assert(false);
    
    return NULL;
}

void zscore(vector< vector<double>* >* imageDescriptions, string saveFileName)
{
    vector<double> mean;
    vector<double> stdDev;
    assert(imageDescriptions->size() > 0);
    mean.resize(imageDescriptions->front()->size(),0);
    for (vector<double>* image : *imageDescriptions)
    {
        for (int i=0; i<mean.size(); i++)
            mean[i] += image->at(i);
    }
    for (int i=0; i<mean.size(); i++)
        mean[i] /= imageDescriptions->size();
    
    stdDev.resize(imageDescriptions->front()->size(),0);
    for (vector<double>* image : *imageDescriptions)
    {
        for (int i=0; i<mean.size(); i++)
            stdDev[i] += pow(image->at(i)-mean[i],2);
    }
    for (int i=0; i<stdDev.size(); i++)
        stdDev[i] = sqrt(stdDev[i]/imageDescriptions->size());
    
    for (vector<double>* image : *imageDescriptions)
    {
        for (int i=0; i<image->size(); i++)
            if (stdDev[i] != 0)
                image->at(i) = (image->at(i)-mean[i])/stdDev[i];
            else
                image->at(i) = (image->at(i)-mean[i]);
    }
    
    ofstream saveZ(saveFileName);
    saveZ << "Mean:";
    for (int i=0; i<mean.size(); i++)
        saveZ <<setprecision(15) << mean[i] << ",";
    saveZ << "\nStdDev:";
    for (int i=0; i<stdDev.size(); i++)
        saveZ <<setprecision(15) << stdDev[i] << ",";
    saveZ << endl;
    saveZ.close();
}

void zscoreUse(vector< vector<double>* >* imageDescriptions, string loadFileName)
{
    vector<double> mean;
    vector<double> stdDev;
    ifstream loadZ(loadFileName);
    if (!loadZ)
    {
	    cout << "ERROR: no zscore saved: " << loadFileName << endl;
	    assert(false);
	            exit(-1);
    }
    regex parse_line("\\w*:(((-?[0-9]*(\\.[0-9]+e?-?[0-9]*)?),)+)");
    regex parse_values("(-?[0-9]*(\\.[0-9]+e?-?[0-9]*)?),");
    string line;
    smatch sm;
    getline(loadZ,line);
    if(regex_search(line,sm,parse_line))
    {
        string values = string(sm[1]);
        while(regex_search(values,sm,parse_values))
        {
            mean.push_back(my_stof(sm[1]));
            values = sm.suffix();
        }
    }
    else
    {
        cout << "ERROR, no Z mean"<<endl;
        assert(false);
        exit(-1);
    }
    
    getline(loadZ,line);
    if(regex_search(line,sm,parse_line))
    {
        string values = string(sm[1]);
        while(regex_search(values,sm,parse_values))
        {
            stdDev.push_back(my_stof(sm[1]));
            values = sm.suffix();
        }
    }
    else
    {
        cout << "ERROR, no Z stdDev"<<endl;
        assert(false);
        exit(-1);
    }
    
    for (vector<double>* image : *imageDescriptions)
    {
        for (int i=0; i<image->size(); i++)
            if (stdDev[i] != 0)
                image->at(i) = (image->at(i)-mean.at(i))/stdDev.at(i);
            else
                image->at(i) = (image->at(i)-mean[i]);
    }
}

void getImageDescriptions(string imageDir, bool trainImages, int positiveClass, const Codebook* codebook, string keyPoint_option, string descriptor_option, string pool_option, string codebookLoc, vector< vector<double>* >* imageDescriptions, vector<double>* imageLabels)
{
    string saveFileName=SAVE_LOC+"imageDesc_"+(trainImages?string("train"):string("test"))+"_"+keyPoint_option+"_"+descriptor_option+"_"+pool_option+"_"+codebookLoc+".save";
    string z_saveFileName=SAVE_LOC+"zscore_"+keyPoint_option+"_"+descriptor_option+"_"+pool_option+"_"+codebookLoc+".save";
    ifstream load(saveFileName);
    if (!load)
    {
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (imageDir.c_str())) == NULL)
        {
            cout << "ERROR: failed to open image directory" << endl;
            exit(-1);
        }
         cout << "reading images and obtaining descriptions" << endl;
          
        vector<string> fileNames;
	//vector<string> maskFileNames;
        while ((ent = readdir (dir)) != NULL) {
          string fileName(ent->d_name);
          smatch sm;
          regex parse("([0-9][0-9][0-9])_([0-9][0-9][0-9][0-9]).jpg");
          if (regex_search(fileName,sm,parse))
          {
              if ((trainImages && isTrainingImage(stoi(sm[2]))) || 
                  (!trainImages && isTestingImage(stoi(sm[2]))) )
              {
                fileNames.push_back(fileName);
                //maskFileNames.push_back(string(sm[1])+"_"+string(sm[2])+"_mask.png");

                imageLabels->push_back(stoi(sm[1]));
              }
          }
        }
        
        unsigned int loopCrit = fileNames.size();
        imageDescriptions->resize(fileNames.size());
        #pragma omp parallel for num_threads(3)
        for (unsigned int nameIdx=0; nameIdx<loopCrit; nameIdx++)
        {
              
              
            string fileName=fileNames[nameIdx];

            Mat color_img = imread(imageDir+fileName, CV_LOAD_IMAGE_COLOR);
	    //Mat mask = imread(DEFAULT_MASK_DIR+maskFileNames[nameIdx], CV_LOAD_IMAGE_GRAY);

            vector<KeyPoint>* keyPoints = getKeyPoints(keyPoint_option,color_img);
            vector< vector<double> >* descriptors = getDescriptors(descriptor_option,color_img,keyPoints);
            vector<double>* imageDescription = getImageDescription(pool_option,color_img,keyPoints,descriptors,codebook);
            assert(imageDescription != NULL);
            #pragma omp critical
            {
                imageDescriptions->at(nameIdx) = (imageDescription);
            }

            delete keyPoints;
            delete descriptors;
        }
        
        if (trainImages)
        {
            zscore(imageDescriptions,z_saveFileName);
        }
        else
        {
            zscoreUse(imageDescriptions,z_saveFileName);
        }
        
        //save
        ofstream save(saveFileName);
	assert(save);
        for (unsigned int i=0; i<imageDescriptions->size(); i++)
        {
            save << "Label:" << imageLabels->at(i) << " Desc:";
	    assert(imageDescriptions->at(i)->size()%codebook->size()==0);
            for (unsigned int j=0; j<imageDescriptions->at(i)->size(); j++)
            {
                assert(imageDescriptions->at(i)->at(j) == imageDescriptions->at(i)->at(j));
                save <<setprecision(15)<< imageDescriptions->at(i)->at(j) << ",";
            }
            save << endl;
        }
        save.close();
    }
    else //read in
    {
        string line;
        smatch sm;
        regex parse_option("Label:([0-9]+) Desc:(((-?[0-9]*(\\.[0-9]+e?-?[0-9]*)?),)+)");
        regex parse_values("(-?[0-9]*(\\.[0-9]+e?-?[0-9]*)?),");
        while (getline(load,line))
        {
            
            if(regex_search(line,sm,parse_option))
            {
                //for (unsigned int i=0; i<sm.size(); i+=1)
                //{
                //    cout <<sm[i]<<endl;
                //}
                
                int label = stoi(sm[1]);
                //cout << label << endl;
                imageLabels->push_back(label);
                
                vector<double>* imageDescription = new vector<double>();
                string values = string(sm[2]);
                //cout << "values" << endl;
                while(regex_search(values,sm,parse_values))
                {
                    //cout <<sm[1]<<endl;
                    imageDescription->push_back(my_stof(sm[1]));
                    values = sm.suffix();
                }
		assert(codebook==NULL || imageDescription->size()%codebook->size()==0);
                imageDescriptions->push_back(imageDescription);
            }
        }
        load.close();
        assert(imageDescriptions->size() > 0 );
    }
    
    if (positiveClass>=0)
    {
        for (int i=0; i<imageLabels->size(); i++)
        {
            imageLabels->at(i) = imageLabels->at(i)==positiveClass?1:-1;
        }
    }
}


void train(string imageDir, int positiveClass, const Codebook* codebook, string keypoint_option, string descriptor_option, string pool_option, string codebookLoc, double eps, double C, string modelLoc)
{
    modelLoc += ".svm";
    vector< vector<double>* > imageDescriptions;
    vector<double> imageLabels;
    getImageDescriptions(imageDir, true, positiveClass, codebook, keypoint_option, descriptor_option, pool_option, codebookLoc, &imageDescriptions, &imageLabels);
    
    
    struct svm_problem* prob = new struct svm_problem();
    prob->l = imageDescriptions.size();
    prob->y = imageLabels.data();
    prob->x = new struct svm_node*[imageDescriptions.size()];
    for (unsigned int i=0; i<imageDescriptions.size(); i++)
    {
        prob->x[i] = convertDescription(imageDescriptions[i]);
        delete imageDescriptions[i];
    }
    
    struct svm_parameter* para = new struct svm_parameter();
    if (positiveClass == -1)
    {
        para->svm_type = C_SVC;
        para->nr_weight=0;
        para->weight_label=NULL;
        para->weight=NULL;
    }
    else
    {
        //para->svm_type = ONE_CLASS;
        //para->nu = 0.5;
        para->svm_type = C_SVC;
        para->nr_weight=1;
        para->weight_label=new int[1];
        para->weight_label[0]=1;
        para->weight=new double[1];
        para->weight[0]=9.0;
        para->weight[1]=1.0;
    }
    para->kernel_type = LINEAR;
    if (codebook != NULL)
        para->gamma = 1.0/codebook->size();
    else
        para->gamma = 1.0/imageDescriptions[0]->size();
    
    para->cache_size = 100;
    para->eps = eps;
    para->C = C;
    para->shrinking = 1;
    para->probability = 0;
    
    ///probably not needed, but jic
    para->degree = 3;
    para->coef0 = 0;
    para->nu = 0.5;
    para->p = 0.1;
    /////
    
    const char *err = svm_check_parameter(prob,para);
    if (err!=NULL)
    {
        cout << "ERROR: " << string(err) << endl;
        exit(-1);
    }
    struct svm_model *trained_model = svm_train(prob,para);
    
    
    int err2 = svm_save_model(modelLoc.c_str(), trained_model);
    
    if (err2 == -1)
    {
        cout << "ERROR: failed to save model" << endl;
        exit(-1);
    }
    cout << "saved as " << modelLoc << endl;
    
    int numLabels = svm_get_nr_class(trained_model);
    double* dec_values = new double[numLabels*(numLabels-1)/2];
    
    int correct = 0;
    int found =0;
    int ret=0;
    int numP=0;
    for (unsigned int i=0; i<imageDescriptions.size(); i++)
    {
        struct svm_node* x = prob->x[i];
        double class_prediction = svm_predict_values(trained_model, x, dec_values);
        if (class_prediction == imageLabels[i])
            correct++;
            
        if (positiveClass != -1)
        {    
            if ((imageLabels[i]==1) && class_prediction>0)
                    found++;
                
            if (imageLabels[i]==1)
                numP++;
                
            if (class_prediction>0)
                ret++;
        }
    }
    cout << "Accuracy on training data: " << correct/(double)imageDescriptions.size() << endl;
    if (positiveClass != -1)
    {
        cout << "recall: " << found/(double)numP << endl;
        cout << "precision: " << found/(double)ret << endl;
    }
    svm_free_model_content(trained_model);
    svm_destroy_param(para);
    delete[] dec_values;
}


void trainSVM_CV(string imageDir, int positiveClass, const Codebook* codebook, string keypoint_option, string descriptor_option, string pool_option, string codebookLoc, double eps, double C, string modelLoc)
{
    modelLoc += ".xml";
    vector< vector<double>* > imageDescriptions;
    vector<double> imageLabels;
    getImageDescriptions(imageDir, true, positiveClass, codebook, keypoint_option, descriptor_option, pool_option, codebookLoc, &imageDescriptions, &imageLabels);
    

    
    Mat labelsMat(imageLabels.size(), 1, CV_32FC1);
    for (int j=0; j<imageLabels.size(); j++)
    {
        labelsMat.at<float>(j,0) = imageLabels[j];
    }

    
    Mat trainingDataMat(imageDescriptions.size(), imageDescriptions[0]->size(), CV_32FC1);
    for (int j=0; j<imageDescriptions.size(); j++)
        for (int i=0; i<imageDescriptions[j]->size(); i++)
        {
            trainingDataMat.at<float>(j,i)=imageDescriptions[j]->at(i);
        }
    
    CvSVMParams params;
    //params.svm_type    = CvSVM::C_SVC;
    //params.kernel_type = CvSVM::LINEAR;
    //params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    Mat weighting = (Mat_<float>(2,1) << 9.0,1.0);
    CvMat www = weighting;
    if (positiveClass != -1)
    {
        
        params.class_weights = &www;
    }

    CvSVM SVM;
    SVM.train_auto(trainingDataMat, labelsMat, Mat(), Mat(), params);
    
    
    SVM.save(modelLoc.c_str());
    cout << "saved as " << modelLoc << endl;
    
    
    int correct = 0;
    int found =0;
    int ret=0;
    int numP=0;
    for (unsigned int j=0; j<imageDescriptions.size(); j++)
    {
        
        Mat instance(1, imageDescriptions[j]->size(), CV_32FC1);
        for (int i=0; i<imageDescriptions[j]->size(); i++)
        {
            instance.at<float>(0,i)=imageDescriptions[j]->at(i);
        }
        
        float class_prediction = SVM.predict(instance,false);
        if (class_prediction == imageLabels[j])
            correct++;
            
        if (positiveClass != -1)
        {    
            if ((imageLabels[j]==1) && class_prediction>0)
                    found++;
                
            if (imageLabels[j]==1)
                numP++;
                
            if (class_prediction>0)
                ret++;
        }
    }
    cout << "Accuracy on training data: " << correct/(double)imageDescriptions.size() << endl;
    if (positiveClass != -1)
    {
        cout << "recall: " << found/(double)numP << endl;
        cout << "precision: " << found/(double)ret << endl;
    }
}
///////////////////////////////////////////
//#include "tester.cpp"
void test()
{
    
    assert(hasPrefix("abcd","a"));
    assert(hasPrefix("abcd","abcd"));
    assert(!hasPrefix("abcd","bcd"));
    assert(!hasPrefix("abcd","abcde"));
    assert(!hasPrefix(" abcd","a"));
    
    assert(isTrainingImage(1));
    assert(!isTrainingImage(2));
    assert(!isTestingImage(1));
    assert(isTestingImage(2));
    
    assert(0.0 == my_stof(".0"));
    assert(0.0 == my_stof("0.0"));
    assert(0.0 == my_stof("0"));
    assert(10.0 == my_stof("10.0"));
    assert(100.0 == my_stof("1.0e2"));
    assert(0.0001 == my_stof("1.0e-4"));
    
    vector<double> desc0 = {1.0,2.0,3.0};
    struct svm_node* test_node0 = convertDescription(&desc0);
    assert(test_node0[0].value == 1.0);
    assert(test_node0[1].value == 2.0);
    assert(test_node0[2].value == 3.0);
    assert(test_node0[0].index == 0);
    assert(test_node0[1].index == 1);
    assert(test_node0[2].index == 2);
    assert(test_node0[3].index == -1);
    
    vector<double> desc1 = {1.0,0.0,3.0};
    struct svm_node* test_node1 = convertDescription(&desc1);
    assert(test_node1[0].value == 1.0);
    assert(test_node1[1].value == 3.0);
    assert(test_node1[0].index == 0);
    assert(test_node1[1].index == 2);
    assert(test_node1[2].index == -1);
    
    vector< vector<double>* > vec0(8);
    vec0[0] = new vector<double>(3);
    vec0[0]->at(0) = 2;
    vec0[0]->at(1) = 2;
    vec0[0]->at(2) = 3;
    vec0[1] = new vector<double>(3);
    vec0[1]->at(0) = 4;
    vec0[1]->at(1) = 2.2;
    vec0[1]->at(2) = 3;
    vec0[2] = new vector<double>(3);
    vec0[2]->at(0) = 4;
    vec0[2]->at(1) = 2;
    vec0[2]->at(2) = 3;
    vec0[3] = new vector<double>(3);
    vec0[3]->at(0) = 4;
    vec0[3]->at(1) = 2.2;
    vec0[3]->at(2) = 3;
    vec0[4] = new vector<double>(3);
    vec0[4]->at(0) = 5;
    vec0[4]->at(1) = 2;
    vec0[4]->at(2) = 3;
    vec0[5] = new vector<double>(3);
    vec0[5]->at(0) = 5;
    vec0[5]->at(1) = 2;
    vec0[5]->at(2) = 3;
    vec0[6] = new vector<double>(3);
    vec0[6]->at(0) = 7;
    vec0[6]->at(1) = 2.4;
    vec0[6]->at(2) = 3;
    vec0[7] = new vector<double>(3);
    vec0[7]->at(0) = 9;
    vec0[7]->at(1) = 2.1;
    vec0[7]->at(2) = 3;
    
    
    vector< vector<double>* > vec1(8);
    vec1[0] = new vector<double>(3);
    vec1[0]->at(0) = 2;
    vec1[0]->at(1) = 2;
    vec1[0]->at(2) = 3;
    vec1[1] = new vector<double>(3);
    vec1[1]->at(0) = 4;
    vec1[1]->at(1) = 2.2;
    vec1[1]->at(2) = 3;
    vec1[2] = new vector<double>(3);
    vec1[2]->at(0) = 4;
    vec1[2]->at(1) = 2;
    vec1[2]->at(2) = 3;
    vec1[3] = new vector<double>(3);
    vec1[3]->at(0) = 4;
    vec1[3]->at(1) = 2.2;
    vec1[3]->at(2) = 3;
    vec1[4] = new vector<double>(3);
    vec1[4]->at(0) = 5;
    vec1[4]->at(1) = 2;
    vec1[4]->at(2) = 3;
    vec1[5] = new vector<double>(3);
    vec1[5]->at(0) = 5;
    vec1[5]->at(1) = 2;
    vec1[5]->at(2) = 3;
    vec1[6] = new vector<double>(3);
    vec1[6]->at(0) = 7;
    vec1[6]->at(1) = 2.4;
    vec1[6]->at(2) = 3;
    vec1[7] = new vector<double>(3);
    vec1[7]->at(0) = 9;
    vec1[7]->at(1) = 2.1;
    vec1[7]->at(2) = 3;
    
    zscore(&vec0,"save/temp");
    zscoreUse(&vec1,"save/temp");
    assert(vec0[0]->at(0)==(2-5.0)/2.0);
    assert(vec0[1]->at(0)==(4-5.0)/2.0);
    assert(vec0[1]->at(2)==0.0);
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            assert(fabs(vec0[i]->at(j) - vec1[i]->at(j))<.0001);
            

}
////////////////

int main(int argc, char** argv)
{
    vector<int> labelsGlobal={1,2,3,4,5,6,7,8,9,10};
    string option = argv[1];
    if (hasPrefix(option,"compare_SIFT"))
    {
        Mat img = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
        assert(img.rows!=0);
        int minHessian = 400;
        int nfeaturePoints=10;
        int nOctivesPerLayer=3;
        SIFT detector(nfeaturePoints,nOctivesPerLayer);
        vector<KeyPoint> kps;
        Mat desc;
        detector(img,noArray(),kps,desc);
        cout << desc.depth() << endl;
        
        
        vector< vector<double> > descriptions;
        CustomSIFT::Instance()->extract(img,kps,descriptions);
        
        
        for (unsigned int i=0; i<kps.size(); i++)
        {
            double sum=0;
            for (unsigned int j=0; j<128; j++)
            {
                sum += desc.at<float>(i,j)*desc.at<float>(i,j);
            }
            double norm = sqrt(sum);
            for (unsigned int j=0; j<128; j++)
            {
                 desc.at<float>(i,j) /= norm;
            }
        }
        
        for (unsigned int i=0; i<kps.size(); i++)
        {
            assert(desc.cols==128);
            assert(descriptions[i].size()==128);
            for (unsigned int j=0; j<128; j++)
            {
                if (desc.at<float>(i,j) != descriptions[i][j])
                    cout << "["<<i<<"]["<<j<<"] "<<desc.at<float>(i,j)<<"\t"<<descriptions[i][j]<<endl;
            }
        }
    }
    else if (hasPrefix(option,"train_codebook"))
    {
        int codebook_size=DEFAULT_CODEBOOK_SIZE;
        smatch sm_option;
        regex parse_option("train_codebook_size=([0-9]+)");
        if(regex_search(option,sm_option,parse_option))
        {
            codebook_size = stoi(sm_option[1]);
            cout << "codebook size = " << codebook_size << endl;
        }
        
        string keypoint_option = argv[2];
        string descriptor_option = argv[3];
        
        string imageDir = argv[argc-2];
        if (imageDir == "default") imageDir = DEFAULT_IMG_DIR;
        if (imageDir[imageDir.size()-1]!='/') imageDir += '/';
        string codebookLoc = argv[argc-1];
        if (codebookLoc == "default") codebookLoc = SAVE_LOC+"codebook_"+to_string(codebook_size)+"_"+keypoint_option+"_"+descriptor_option+".cb";
        
        vector< vector<double> > accum;
        
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (imageDir.c_str())) != NULL)
        {
          cout << "reading images and obtaining descriptors" << endl;
          
          vector<string> fileNames;
          while ((ent = readdir (dir)) != NULL) {
              string fileName(ent->d_name);
              if (fileName[0] == '.' || (fileName[fileName.size()-1]!='G' && fileName[fileName.size()-1]!='g' &&  fileName[fileName.size()-1]!='f'))
                  continue;
              fileNames.push_back(fileName);
          }
          
          int loopCrit = min((int)(300*codebook_size/50),(int)fileNames.size());
          //int loopCrit = min((int)(300),(int)fileNames.size());
    #pragma omp parallel for num_threads(3)
          for (unsigned int nameIdx=0; nameIdx<loopCrit; nameIdx++)
          {
              
              
              string fileName=fileNames[nameIdx];
              
              Mat color_img = imread(imageDir+fileName, CV_LOAD_IMAGE_COLOR);
              
              vector<KeyPoint>* keyPoints = getKeyPoints(keypoint_option,color_img);
              vector< vector<double> >* descriptors = getDescriptors(descriptor_option,color_img,keyPoints);
              
    #pragma omp critical
              {
                  for (const vector<double>& description : *descriptors)
                  {
                      if (description.size() > 0)
                          accum.push_back(description);
                  }
              }
              
              delete keyPoints;
              delete descriptors;
          }
          
          Codebook codebook;
          codebook.trainFromExamples(codebook_size,accum);
          
          codebook.save(codebookLoc);
        }
        else
            cout << "Error, could not load files for codebook." << endl;
    }
    else if (hasPrefix(option,"train_libsvm"))//train_svm_eps=$D_C=$D_AllVs=$POSITIVECLASS $CODEBOOKLOC $MODELLOC)
    {
        double eps = 0.001;
        double C = 2.0;
        smatch sm_option;
        int positiveClass = -1;
        regex parse_option("train_libsvm_eps=(-?[0-9]*(\\.[0-9]+e?-?[0-9]*)?)_C=(-?[0-9]*(\\.[0-9]+e?-?[0-9]*)?)_AllVs=(-?[0-9]+)");
        if(regex_search(option,sm_option,parse_option))
        {
            eps = my_stof(sm_option[1]);
            C = my_stof(sm_option[3]);
            positiveClass = stoi(sm_option[5]);
            cout << "eps="<<eps<<" C="<<C<<" positiveClass="<<positiveClass<<endl;
        }
        
        
        string keypoint_option = argv[2];
        string descriptor_option = argv[3];
        string pool_option = argv[4];
        
        string imageDir = argv[argc-3];
        if (imageDir == "default") imageDir = DEFAULT_IMG_DIR;
        if (imageDir[imageDir.size()-1]!='/') imageDir += '/';
        
        string codebookLoc = argv[argc-2];
        if (codebookLoc == "default") codebookLoc = SAVE_LOC+"codebook_"+to_string(DEFAULT_CODEBOOK_SIZE)+"_"+keypoint_option+"_"+descriptor_option+".cb";
        string modelLoc = argv[argc-1];
        
        
        Codebook codebook;
        if (codebookLoc != "none"){
	    codebook.readIn(codebookLoc);
	    codebookLoc=codebookLoc.substr(codebookLoc.find_last_of('/')+1);
	}
	else     codebookLoc="";


	if (modelLoc == "default") 
		        if (positiveClass != -2)
					                        modelLoc = SAVE_LOC+"model_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+"_"+codebookLoc+"_"+to_string(positiveClass);
	                else
				                        modelLoc = SAVE_LOC+"model_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+"_"+codebookLoc+"_";

        if (positiveClass != -2)
            train(imageDir, positiveClass, (codebookLoc != "")?&codebook:NULL, keypoint_option, descriptor_option, pool_option, codebookLoc, eps, C, modelLoc);
        else
        {
            
            
            
            for (int label : labelsGlobal)
            {
                train(imageDir, label, &codebook, keypoint_option, descriptor_option, pool_option, codebookLoc, eps, C, modelLoc+to_string(label));
            }
        }
    }
    else if (hasPrefix(option,"train_cvsvm"))//train_svm_eps=$D_C=$D_AllVs=$POSITIVECLASS $CODEBOOKLOC $MODELLOC)
    {
        double eps = 0.001;
        double C = 2.0;
        smatch sm_option;
        int positiveClass = -1;
        regex parse_option("train_cvsvm_AllVs=(-?[0-9]+)");
        if(regex_search(option,sm_option,parse_option))
        {
            positiveClass = stoi(sm_option[1]);
            cout << "positiveClass="<<positiveClass<<endl;
        }
        
        
        string keypoint_option = argv[2];
        string descriptor_option = argv[3];
        string pool_option = argv[4];
        
        string imageDir = argv[argc-3];
        if (imageDir == "default") imageDir = DEFAULT_IMG_DIR;
        if (imageDir[imageDir.size()-1]!='/') imageDir += '/';
        
        string codebookLoc = argv[argc-2];
        if (codebookLoc == "default") codebookLoc = SAVE_LOC+"codebook_"+to_string(DEFAULT_CODEBOOK_SIZE)+"_"+keypoint_option+"_"+descriptor_option+".cb";
        string modelLoc = argv[argc-1];
        
        
        Codebook codebook;
        if (codebookLoc != "none"){
	    codebook.readIn(codebookLoc);
	    codebookLoc=codebookLoc.substr(codebookLoc.find_last_of('/')+1);
	}
	else     codebookLoc="";
        

        if (modelLoc == "default") 
		if (positiveClass != -2)
			modelLoc = SAVE_LOC+"model_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+"_"+codebookLoc+"_"+to_string(positiveClass);
		else
			modelLoc = SAVE_LOC+"model_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+"_"+codebookLoc+"_";


        if (positiveClass != -2)
            trainSVM_CV(imageDir, positiveClass, (codebookLoc != "")?&codebook:NULL, keypoint_option, descriptor_option, pool_option, codebookLoc, eps, C, modelLoc);
        else
        {
            
            
            
            for (int label : labelsGlobal)
            {
                trainSVM_CV(imageDir, label, &codebook, keypoint_option, descriptor_option, pool_option, codebookLoc, eps, C, modelLoc+to_string(label));
            }
        }
        ///////////////////////
        cout << "Done\nProcessing test images" << endl;
        vector< vector<double>* > imageDescriptions;
        vector<double> imageLabels;
        getImageDescriptions(imageDir, false, positiveClass, (codebookLoc != "none")?&codebook:NULL, keypoint_option, descriptor_option, pool_option, codebookLoc, &imageDescriptions, &imageLabels);
        /////////////
    }
    else if (hasPrefix(option,"test_libsvm"))
    {
        smatch sm_option;
        int positiveClass = -1;
        regex parse_option("test_libsvm_AllVs=(-?[0-9]+)");
        if(regex_search(option,sm_option,parse_option))
        {
            positiveClass = stoi(sm_option[1]);
            cout << "positiveClass="<<positiveClass<<endl;
        }
        
        string keypoint_option = argv[2];
        string descriptor_option = argv[3];
        string pool_option = argv[4];
        
        string imageDir = argv[argc-3];
        if (imageDir == "default")
            imageDir = DEFAULT_IMG_DIR;
        if (imageDir[imageDir.size()-1]!='/')
            imageDir += '/';
        string codebookLoc = argv[argc-2];
        if (codebookLoc == "default")
            codebookLoc = SAVE_LOC+"codebook_"+to_string(DEFAULT_CODEBOOK_SIZE)+"_"+keypoint_option+"_"+descriptor_option+".cb";
        string modelLoc = argv[argc-1];
        
        
        
        Codebook codebook;
        if (codebookLoc != "none"){
	    codebook.readIn(codebookLoc);
	    codebookLoc=codebookLoc.substr(codebookLoc.find_last_of('/')+1);
	}
	else     codebookLoc="";
        vector< vector<double>* > imageDescriptions;
        vector<double> imageLabels;
        
        
        getImageDescriptions(imageDir, false, positiveClass, (codebookLoc != "none")?&codebook:NULL, keypoint_option, descriptor_option, pool_option, codebookLoc, &imageDescriptions, &imageLabels);
        
        
        
        if (positiveClass != -2)
        {
            if (modelLoc == "default")
                modelLoc = SAVE_LOC+"model_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+"_"+codebookLoc+"_"+to_string(positiveClass)+".svm";
        
            struct svm_model* trained_model = svm_load_model(modelLoc.c_str());
            
            int numLabels = svm_get_nr_class(trained_model);
            int* labels = new int[numLabels];
            svm_get_labels(trained_model, labels);
            double* dec_values = new double[numLabels*(numLabels-1)/2];
            
            int correct = 0;
            int found =0;
            int ret=0;
            int numP=0;
            for (unsigned int i=0; i<imageDescriptions.size(); i++)
            {
                struct svm_node* x = convertDescription(imageDescriptions[i]);
                double class_prediction = svm_predict_values(trained_model, x, dec_values);
                if (class_prediction == imageLabels[i])
                    correct++;
                
                if (positiveClass != -1)
                {    
                    if ((imageLabels[i]==1) && class_prediction>0)
                            found++;
                        
                    if (imageLabels[i]==1)
                        numP++;
                        
                    if (class_prediction>0)
                        ret++;
                } 
                delete imageDescriptions[i];
                delete x;
            }
            cout << "Accuracy: " << correct/(double)imageDescriptions.size() << endl;
            if (positiveClass != -1)
            {
                cout << "recall: " << found/(double)numP << endl;
                cout << "precision: " << found/(double)ret << endl;
            }
            svm_free_model_content(trained_model);
            delete[] labels;
        }
        else////train 10 modesl
        {
            map<int,struct svm_model*> trained_models;
            for (int label : labelsGlobal)
            {
                string thisModelLoc;
                if (modelLoc == "default")
                    thisModelLoc = SAVE_LOC+"model_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+"_"+codebookLoc+"_"+to_string(label)+".svm";
                else
                    thisModelLoc = modelLoc + to_string(label);
                trained_models[label] = svm_load_model(thisModelLoc.c_str());
                assert(trained_models[label] != NULL);
                 cout << "loaded " << thisModelLoc << endl;
            }
            
            ////////
            ///////
            int numLabels = svm_get_nr_class(trained_models[1]);
            cout << "num labels " << numLabels << endl;
            int* labels = new int[numLabels];
            svm_get_labels(trained_models[1], labels);
            double* dec_values = new double[numLabels*(numLabels-1)/2];
            int correct = 0;
            for (unsigned int i=0; i<imageDescriptions.size(); i++)
            {
                
                double class_prediction=0;
                double conf=0;
                for (int label : labelsGlobal)
                {
                    struct svm_node* x = convertDescription(imageDescriptions[i]);
                    double is_this_class = svm_predict_values(trained_models[label], x, dec_values);
                    if (is_this_class > 0 && dec_values[0]>conf)
                    {
                        class_prediction = label;
                        conf = dec_values[0];
                    }
                    //cout <<i << ", for label " << label << " : " << is_this_class << endl;
                    //for (unsigned int d=0; d<numLabels*(numLabels-1)/2; d++)
                    //    cout << dec_values[d] << ", ";
                    //cout << endl;
                    delete x;
                }
                //cout << "actual : " << imageLabels[i] << endl;
                if (class_prediction == imageLabels[i])
                    correct++;
                delete imageDescriptions[i];
                
            }
            cout << "Accuracy: " << correct/(double)imageDescriptions.size() << endl;
            for (int label : labelsGlobal)
            {
                svm_free_model_content(trained_models[label]);
            }
            delete[] labels;
        }
        
        
        
    }
    else if (hasPrefix(option,"test_cvsvm"))
    {
        smatch sm_option;
        int positiveClass = -1;
        bool drawPR=false;
        regex parse_option("AllVs=(-?[0-9]+)");
        if(regex_search(option,sm_option,parse_option))
        {
            positiveClass = stoi(sm_option[1]);
            cout << "positiveClass="<<positiveClass<<endl;
        }
        
        regex parse_option_pr("[pP][rR](curves?)?");
        if(regex_search(option,sm_option,parse_option_pr))
        {
            drawPR=true;
            cout << "PR curves!" <<endl;
        }
        
        string keypoint_option = argv[2];
        string descriptor_option = argv[3];
        string pool_option = argv[4];
        
        string imageDir = argv[argc-3];
        if (imageDir == "default")
            imageDir = DEFAULT_IMG_DIR;
        if (imageDir[imageDir.size()-1]!='/')
            imageDir += '/';
        string codebookLoc = argv[argc-2];
        if (codebookLoc == "default")
            codebookLoc = SAVE_LOC+"codebook_"+to_string(DEFAULT_CODEBOOK_SIZE)+"_"+keypoint_option+"_"+descriptor_option+".cb";
        string modelLoc = argv[argc-1];
        
        
        
        Codebook codebook;
        if (codebookLoc != "none"){
	    codebook.readIn(codebookLoc);
	    codebookLoc=codebookLoc.substr(codebookLoc.find_last_of('/')+1);
	}
	else     codebookLoc="";
        vector< vector<double>* > imageDescriptions;
        vector<double> imageLabels;
        
        
        getImageDescriptions(imageDir, false, positiveClass, (codebookLoc != "none")?&codebook:NULL, keypoint_option, descriptor_option, pool_option, codebookLoc, &imageDescriptions, &imageLabels);
        
        
        
        if (positiveClass != -2)
        {
            if (modelLoc == "default")
                modelLoc = SAVE_LOC+"model_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+"_"+codebookLoc+"_"+to_string(positiveClass)+".xml";
            
            CvSVM SVM;
            SVM.load(modelLoc.c_str());
            
            
            int correct = 0;
            int found =0;
            int ret=0;
            int numP=0;
            
            #if DEBUG_SHOW
            map<int, vector< vector<double>* > > byClass;
            vector<double> maxs = *imageDescriptions[0];
            vector<double> mins = *imageDescriptions[0];
            #endif
            
            for (unsigned int j=0; j<imageDescriptions.size(); j++)
            {
                assert(codebookLoc == "none" || imageDescriptions[j]->size() == codebook.size());
                Mat instance(1, imageDescriptions[j]->size(), CV_32FC1);
                for (int i=0; i<imageDescriptions[j]->size(); i++)
                {
                    instance.at<float>(0,i)=imageDescriptions[j]->at(i);
                }
                float class_prediction = SVM.predict(instance,false);
                if (class_prediction == imageLabels[j])
                    correct++;
                    
                if (positiveClass != -1)
                {    
                    if ((imageLabels[j]==1) && class_prediction>0)
                            found++;
                        
                    if (imageLabels[j]==1)
                        numP++;
                        
                    if (class_prediction>0)
                        ret++;
                }
                
                #if DEBUG_SHOW
                byClass[imageLabels[j]].push_back(imageDescriptions[j]);
                for (int ii=0; ii<imageDescriptions[j]->size(); ii++)
                {
                    if (imageDescriptions[j]->at(ii) > maxs[ii])
                        maxs[ii]=imageDescriptions[j]->at(ii);
                    if (imageDescriptions[j]->at(ii) < mins[ii])
                        mins[ii]=imageDescriptions[j]->at(ii);
                }
                #endif
            }
            cout << "Accuracy: " << correct/(double)imageDescriptions.size() << endl;
            if (positiveClass != -1)
            {
                cout << "recall: " << found/(double)numP << endl;
                cout << "precision: " << found/(double)ret << endl;
            }
            
            #if DEBUG_SHOW
            int row=0;
            Mat full(imageDescriptions.size()*2,imageDescriptions[0]->size()*3,CV_8UC3);
            for (auto p : byClass)
            {
                for (auto ins : p.second)
                {
                    makeHistFull(*ins,maxs,mins,full,row);
                    assert(row<imageDescriptions.size()*2);
                    row+=2;
                }
            }
            imwrite("hist.png",full);
            imshow("hist",full);
            waitKey();
            waitKey();
            #endif
        }
        else////test 10 models
        {
            map<int,CvSVM> trained_models;
            map<int, vector<tuple<float,bool> > > models_PRData;
            for (int label : labelsGlobal)
            {
                string thisModelLoc;
                if (modelLoc == "default")
                    thisModelLoc = SAVE_LOC+"model_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+"_"+codebookLoc+"_"+to_string(label)+".xml";
                else
                    thisModelLoc = modelLoc + to_string(label);
                trained_models[label].load(thisModelLoc.c_str());
                //if (loaded)
                //    cout << "loaded " << thisModelLoc << endl;
                //else
                //    cout << "failed to load " << thisModelLoc << endl;
            }
            
            
            int correct = 0;
            for (unsigned int j=0; j<imageDescriptions.size(); j++)
            {
                
                double class_prediction=0;
                double conf=0;
                Mat instance(1, imageDescriptions[j]->size(), CV_32FC1);
                for (int i=0; i<imageDescriptions[j]->size(); i++)
                {
                    instance.at<float>(0,i)=imageDescriptions[j]->at(i);
                }
                for (int label : labelsGlobal)
                {
                    float is_this_class = trained_models[label].predict(instance,true);
                    models_PRData[label].push_back(make_tuple(is_this_class,imageLabels[j]==label));
                    if (is_this_class < 0 && is_this_class<conf)
                    {
                        class_prediction = label;
                        conf = is_this_class;
                    }
                }
                //cout << "actual : " << imageLabels[i] << endl;
                if (class_prediction == imageLabels[j])
                    correct++;
                delete imageDescriptions[j];
                
            }
            cout << "Accuracy: " << correct/(double)imageDescriptions.size() << endl;
            
            //Generate PR curves
            map<int,double> aps;
            if (drawPR)
            {
                double mAP=0;
                for (int label : labelsGlobal)
                {
                    aps[label] = drawPRCurve(label,models_PRData[label]);
                    mAP += aps[label];
                }
                mAP /= labelsGlobal.size();
                cout << "mAP = " << mAP << endl;
            }
            
        }
        
    }
    else if (hasPrefix(option,"runtest"))
    {
        test();
	    Codebook cb;
	    cb.unittest();
    }
    else
        cout << "ERROR no option: " << option << endl;
    
    return 0;
}



