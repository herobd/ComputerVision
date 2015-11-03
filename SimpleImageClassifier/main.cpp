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

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/ml/ml.hpp"

#include "svm.h"

#include "customsift.h"
#include "codebook.h"

#define NORMALIZE_DESC 0

#define SAVE_LOC string("./save/")
#define DEFAULT_IMG_DIR string("./leedsbutterfly/images/")
#define DEFAULT_CODEBOOK_SIZE 200

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
        return NULL;
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
        
        vector< vector<double> >* ret;// = new vector< vector<double> >(desc.rows);
        CustomSIFT::Instance()->extract(img,*keyPoints,*ret);
        
        return ret;
    }
    
    return NULL;
}

vector<double>* getImageDescription(string option, const vector<KeyPoint>* keyPoints, const vector< vector<double> >* descriptors, const Codebook* codebook)
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
        stdDev[i] = sqrt(stdDev[i]);
    
    for (vector<double>* image : *imageDescriptions)
    {
        for (int i=0; i<image->size(); i++)
            if (stdDev[i] != 0)
                image->at(i) = (image->at(i)-mean[i])/stdDev[i];
    }
    
    ofstream saveZ(saveFileName);
    saveZ << "Mean:";
    for (int i=0; i<mean.size(); i++)
        saveZ << mean[i] << ",";
    saveZ << "\nStdDev:";
    for (int i=0; i<stdDev.size(); i++)
        saveZ << stdDev[i] << ",";
    saveZ << endl;
    saveZ.close();
}

void zscoreUse(vector< vector<double>* >* imageDescriptions, string loadFileName)
{
    ifstream loadZ(loadFileName);
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
    }
}

void getImageDescriptions(string imageDir, bool trainImages, int positiveClass, const Codebook* codebook, string keyPoint_option, string descriptor_option, string pool_option, vector< vector<double>* >* imageDescriptions, vector<double>* imageLabels)
{
    string saveFileName=SAVE_LOC+"imageDesc_"+(trainImages?string("train"):string("test"))+"_"+keyPoint_option+"_"+descriptor_option+"_"+pool_option+".save";
    string z_saveFileName=SAVE_LOC+"zscore_"+keyPoint_option+"_"+descriptor_option+"_"+pool_option+".save";
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
                
                imageLabels->push_back(stoi(sm[1]));
              }
          }
        }
        
        unsigned int loopCrit = fileNames.size();
        #pragma omp parallel for num_threads(3)
        for (unsigned int nameIdx=0; nameIdx<loopCrit; nameIdx++)
        {
              
              
            string fileName=fileNames[nameIdx];

            Mat color_img = imread(imageDir+fileName, CV_LOAD_IMAGE_COLOR);

            vector<KeyPoint>* keyPoints = getKeyPoints(keyPoint_option,color_img);
            vector< vector<double> >* descriptors = getDescriptors(descriptor_option,color_img,keyPoints);
            vector<double>* imageDescription = getImageDescription(pool_option,keyPoints,descriptors,codebook);

            #pragma omp critical
            {
                imageDescriptions->push_back(imageDescription);
            }

            delete keyPoints;
            delete descriptors;
        }
        ;
        if (trainImages)
        {
            zscore(imageDescriptions,z_saveFileName);
        else
        {
            zscoreUse(imageDescriptions,z_saveFileName);
        }
        
        //save
        ofstream save(saveFileName);
        for (unsigned int i=0; i<imageDescriptions->size(); i++)
        {
            save << "Label:" << imageLabels->at(i) << " Desc:";
	    assert(imageDescriptions->at(i)->size() == codebook->size());
            for (unsigned int j=0; j<imageDescriptions->at(i)->size(); j++)
            {
                assert(imageDescriptions->at(i)->at(j) == imageDescriptions->at(i)->at(j));
                save << imageDescriptions->at(i)->at(j) << ",";
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
		assert(codebook==NULL || imageDescription->size() == codebook->size());
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


void train(string imageDir, int positiveClass, const Codebook* codebook, string keypoint_option, string descriptor_option, string pool_option, double eps, double C, string modelLoc)
{
    vector< vector<double>* > imageDescriptions;
    vector<double> imageLabels;
    getImageDescriptions(imageDir, true, positiveClass, codebook, keypoint_option, descriptor_option, pool_option, &imageDescriptions, &imageLabels);
    
    
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
    
    if (modelLoc == "default") modelLoc = SAVE_LOC+"model_"+to_string(positiveClass)+"_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+".svm";
    
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


void trainSVM_CV(string imageDir, int positiveClass, const Codebook* codebook, string keypoint_option, string descriptor_option, string pool_option, double eps, double C, string modelLoc)
{
    vector< vector<double>* > imageDescriptions;
    vector<double> imageLabels;
    getImageDescriptions(imageDir, true, positiveClass, codebook, keypoint_option, descriptor_option, pool_option, &imageDescriptions, &imageLabels);
    

    
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
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    Mat weighting = (Mat_<float>(2,1) << 9.0,1.0);
    CvMat www = weighting;
    if (positiveClass != -1)
    {
       params.class_weights = &www;
    }

    CvSVM SVM;
    SVM.train_auto(trainingDataMat, labelsMat, Mat(), Mat(), params);
    
    if (modelLoc == "default") modelLoc = SAVE_LOC+"model_"+to_string(positiveClass)+"_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+".xml";
    
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
        if (codebookLoc != "none")
            codebook.readIn(codebookLoc);
        
        if (positiveClass != -2)
            train(imageDir, positiveClass, (codebookLoc != "none")?&codebook:NULL, keypoint_option, descriptor_option, pool_option, eps, C, modelLoc);
        else
        {
            
            
            
            for (int label : labelsGlobal)
            {
                train(imageDir, label, &codebook, keypoint_option, descriptor_option, pool_option, eps, C, (modelLoc=="default")?modelLoc:modelLoc+to_string(label));
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
        if (codebookLoc != "none")
            codebook.readIn(codebookLoc);
        
        if (positiveClass != -2)
            trainSVM_CV(imageDir, positiveClass, (codebookLoc != "none")?&codebook:NULL, keypoint_option, descriptor_option, pool_option, eps, C, modelLoc);
        else
        {
            
            
            
            for (int label : labelsGlobal)
            {
                trainSVM_CV(imageDir, label, &codebook, keypoint_option, descriptor_option, pool_option, eps, C, (modelLoc=="default")?modelLoc:modelLoc+to_string(label));
            }
        }
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
        if (codebookLoc != "none")
            codebook.readIn(codebookLoc);
        vector< vector<double>* > imageDescriptions;
        vector<double> imageLabels;
        
        
        getImageDescriptions(imageDir, false, positiveClass, (codebookLoc != "none")?&codebook:NULL, keypoint_option, descriptor_option, pool_option, &imageDescriptions, &imageLabels);
        
        
        
        if (positiveClass != -2)
        {
            if (modelLoc == "default")
                modelLoc = SAVE_LOC+"model_"+to_string(positiveClass)+"_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+".svm";
        
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
        else
        {
            map<int,struct svm_model*> trained_models;
            for (int label : labelsGlobal)
            {
                string thisModelLoc;
                if (modelLoc == "default")
                    thisModelLoc = SAVE_LOC+"model_"+to_string(label)+"_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+".svm";
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
        regex parse_option("test_cvsvm_AllVs=(-?[0-9]+)");
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
        if (codebookLoc != "none")
            codebook.readIn(codebookLoc);
        vector< vector<double>* > imageDescriptions;
        vector<double> imageLabels;
        
        
        getImageDescriptions(imageDir, false, positiveClass, (codebookLoc != "none")?&codebook:NULL, keypoint_option, descriptor_option, pool_option, &imageDescriptions, &imageLabels);
        
        
        
        if (positiveClass != -2)
        {
            if (modelLoc == "default")
                modelLoc = SAVE_LOC+"model_"+to_string(positiveClass)+"_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+".xml";
            
            CvSVM SVM;
            SVM.load(modelLoc.c_str());
            
            
            int correct = 0;
            int found =0;
            int ret=0;
            int numP=0;
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
            }
            cout << "Accuracy: " << correct/(double)imageDescriptions.size() << endl;
            if (positiveClass != -1)
            {
                cout << "recall: " << found/(double)numP << endl;
                cout << "precision: " << found/(double)ret << endl;
            }
        }
        else
        {
            map<int,CvSVM> trained_models;
            for (int label : labelsGlobal)
            {
                string thisModelLoc;
                if (modelLoc == "default")
                    thisModelLoc = SAVE_LOC+"model_"+to_string(label)+"_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+".xml";
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
                    if (is_this_class > 0 && is_this_class>conf)
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
        }
        
    }
    else
        cout << "ERROR no option: " << option << endl;
    
    return 0;
}



