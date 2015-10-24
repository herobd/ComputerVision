/*Brian Daivs
 *CS601R
 *Project one: Features and Learning
 */

#include <iostream>
#include <random>
#include <assert.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include "svm.h"

#include "customsift.h"

using namespace std;
using namespace cv;

bool hasPrefix(const std::string& a, const std::string& pref) {
    return a.substr(0,pref.size()) == b;
}
bool isTrainingImage(int index) {
    return index%2==1;
}
bool isTestingImage(int index) {
    return index%2==0;
}

int main(int argc, char** argv)
{
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
        
        
        for (int i=0; i<kps.size(); i++)
        {
            double sum=0;
            for (int j=0; j<128; j++)
            {
                sum += desc.at<float>(i,j)*desc.at<float>(i,j);
            }
            double norm = sqrt(sum);
            for (int j=0; j<128; j++)
            {
                 desc.at<float>(i,j) /= norm;
            }
        }
        
        for (int i=0; i<kps.size(); i++)
        {
            assert(desc.cols==128);
            assert(descriptions[i].size()==128);
            for (int j=0; j<128; j++)
            {
                if (desc.at<float>(i,j) != descriptions[i][j])
                    cout << "["<<i<<"]["<<j<<"] "<<desc.at<float>(i,j)<<"\t"<<descriptions[i][j]<<endl;
            }
        }
    }
    else if (hasPrefix(option,"train_codebook"))
    {
        string imageDir = argv[argc-2];
        string codebookLoc = argv[argc-1];
        
        
        vector< vector<float> > accum;
        
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
          //private(fileName,img,desc,t)
          int loopCrit = min((int)5000,(int)fileNames.size());
    #pragma omp parallel for 
          for (int nameIdx=0; nameIdx<loopCrit; nameIdx++)
          {
              
              
              string fileName=fileNames[nameIdx];
              
              Mat color_img = imread(directory+fileName, CV_LOAD_IMAGE_COLOR);
              
              vector<KeyPoint>* keyPoints = getKeyPoints(argv[2],color_img);
              vector< vector<double> >* descriptors = getDescriptors(argv[3],color_img,keyPoints);
              
    #pragma omp critical
              {
                  for (cosnt vector<double>& description : *descriptors)
                  {
                      //              assert(get<0>(t).size() > 0);
                      if (description.size() > 0)
                          accum.push_back(description);
                  }
              }
              
              delete keyPoints
              delete descriptors;
          }
          
          Codebook codebook;
          codebook.trainFromExamples(codebook_size,accum);
          
          codebook.save(codebookLoc);
        }
        else
            cout << "Error, could not load files for codebook." << endl;
    }
    else if (hasPrefix(option,"train_svm"))//train_svm_eps=$D_C=$D_AllVs=$POSITIVECLASS $CODEBOOKLOC $MODELLOC)
    {
        double eps = 0.001;
        double C = 2.0;
        smatch sm_option;
        int positiveClass = -1;
        regex parse_option("train_svm_eps=(-?[0-9]*(\\.[0-9]+)?)_C=(-?[0-9]*(\\.[0-9]+)?)_AllVs=([0-9]+)");
        if(regex_search(option,sm_option,parse_option))
        {
            eps = stof(sm_option[1]);
            C = stof(sm_option[2]);
            positiveClass = stoi(sm_option[3]);
        }
        string codebookLoc = argv[argc-2];
        string modelLoc = argv[argc-1];
        
        Codebook codebook;
        codebook.readIn(codebookLoc);
        
        vector< vector<double>* > imageDescriptions;
        vector<double> imageLabels;
        
        
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (imageDir.c_str())) == NULL)
        {
            cout << "ERROR: failed to open training directory" << endl;
            return -1;
        }
         cout << "reading images and obtaining descriptions" << endl;
          
        vector<string> fileNames;
        while ((ent = readdir (dir)) != NULL) {
          string fileName(ent->d_name);
          smatch sm;
          regex parse("([0-9][0-9][0-9])_([0-9][0-9][0-9][0-9]).jpg");
          if (regex_search(fileName,sm,parse))
          {
              if (isTrainingImage(stoi(sm[2]))
              {
                fileNames.push_back(fileName);
                if (positiveClass!=-1)
                    imageLabels.push_back(stoi(sm[1])==positiveClass?1:-1);
                else
                    imageLabels.push_back(stoi(sm[1]));
              }
          }
        }
        //private(fileName,img,desc,t)
        int loopCrit = (int)fileNames.size();
        #pragma omp parallel for 
        for (int nameIdx=0; nameIdx<loopCrit; nameIdx++)
        {
              
              
            string fileName=fileNames[nameIdx];

            Mat color_img = imread(directory+fileName, CV_LOAD_IMAGE_COLOR);

            vector<KeyPoint>* keyPoints = getKeyPoints(argv[2],color_img);
            vector< vector<double> >* descriptors = getDescriptors(argv[3],color_img,keyPoints);
            vector<double>* imageDescription = getImageDescription(argv[4],keyPoints,descriptors,&codebook);

            #pragma omp critical
            {
                imageDescriptions.push_back(imageDescription);
            }

            delete keyPoints
            delete descriptors;
        }
        
        
        
        
        struct svm_problem prob;
        prob.l = imageDescriptions.size();
        prob.y = imageLabels.data();
        prob.x = new struct svm_node*[imageDescriptions.size()];
        for (int i=0; i<imageDescriptions.size(); i++)
        {
            prob.x[i] = convertDescription(imageDescriptions[i]);
            delete imageDescriptions[j];
        }
        
        struct svm_parameter para;
        para.svm_type = C_SVM;
        para.kernel_type = RBF;
        para.gamma = 1.0/codebook.size();
        
        para.cache_size = 100;
        para.eps = eps;
        para.C = C;
        para.nr_weight=0;
        
        const char *err = svm_check_parameter(prob,para);
        if (err!=NULL)
        {
            cout << "ERROR: " << string(err) << endl;
            return -1;
        }
        struct svm_model *trained_model = svm_train(prob,para);
        
        int err2 = svm_save_model(modelLoc, trained_model);
        
        if (err2 == -1)
        {
            cout << "ERROR: failed to save model" << endl;
            return -1;
        }
        
        int correct = 0;
        for (int i=0; i<imageDescriptions.size(); i++)
        {
            svn_node* x = para.x[i];
            double class_prediction = svm_predict_values(trained_model, x);//, double* dec_values
            if (class_prediciton == imageLabels[i])
                correct++;
        }
        cout << "Accuracy on training data: " << correct/(double)imageDescriptions.size() << endl;
        
        svm_free_model_content(trained_model);
        svm_destroy_param(para);
    }
    else if (hasPrefix(option,"test_svm"))
    {
        smatch sm_option;
        int positiveClass = -1;
        regex parse_option("test_svm_AllVs=([0-9]+)");
        if(regex_search(option,sm_option,parse_option))
        {
            positiveClass = stoi(sm_option[1]);
        }
        string codebookLoc = argv[argc-2];
        string modelLoc = argv[argc-1];
        
        struct svm_model* trained_model = svm_load_model(modelLoc.c_str());
        
        
        Codebook codebook;
        codebook.readIn(codebookLoc);
        int positiveClass = atoi(argv[argc-3]);
        vector< vector<double>* > imageDescriptions;
        vector<double> imageLabels;
        
        
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (imageDir.c_str())) == NULL)
        {
            cout << "ERROR: failed to open testing directory" << endl;
            return -1;
        }
         cout << "reading images and obtaining descriptions" << endl;
          
        vector<string> fileNames;
        while ((ent = readdir (dir)) != NULL) {
          string fileName(ent->d_name);
          smatch sm;
          regex parse("([0-9][0-9][0-9])_([0-9][0-9][0-9][0-9]).jpg");
          if (regex_search(fileName,sm,parse))
          {
              if (isTestingImage(stoi(sm[2]))
              {
                fileNames.push_back(fileName);
                if (positiveClass!=-1)
                    imageLabels.push_back(stoi(sm[1])==positiveClass?1:-1);
                else
                    imageLabels.push_back(stoi(sm[1]));
              }
          }
        }
        int loopCrit = (int)fileNames.size();
        #pragma omp parallel for 
        for (int nameIdx=0; nameIdx<loopCrit; nameIdx++)
        {
              
              
            string fileName=fileNames[nameIdx];

            Mat color_img = imread(directory+fileName, CV_LOAD_IMAGE_COLOR);

            vector<KeyPoint>* keyPoints = getKeyPoints(argv[2],color_img);
            vector< vector<double> >* descriptors = getDescriptors(argv[3],color_img,keyPoints);
            vector<double>* imageDescription = getImageDescription(argv[4],keyPoints,descriptors,&codebook);

            #pragma omp critical
            {
                imageDescriptions.push_back(imageDescription);
            }

            delete keyPoints
            delete descriptors;
        }
        
        int correct = 0;
        for (int i=0; i<imageDescriptions.size(); i++)
        {
            svn_node* x = convertDescription(imageDescriptions[i]);
            double class_prediction = svm_predict_values(trained_model, x);//, double* dec_values
            if (class_prediciton == imageLabels[i])
                correct++;
            delete imageDescriptions[i];
        }
        cout << "Accuracy: " << correct/(double)imageDescriptions.size() << endl;
        
        svm_free_model_content(trained_model);
        
    }
    
    return 0;
}


struct svm_node* convertDescription(const vector<double>* description, )
{
    int nonzeroCount=0;
    for (int j=0; j<description->size(); j++)
    {
        if (description->at(i)!=0)
            nonzeroCount++;
    }
    struct svm_node* ret = new struct svm_node[nonzeroCount+1];
    int nonzeroIter=0;
    for (int j=0; j<description.size(); j++)
    {
        if (description->at(i)!=0)
        {
            ret[nonzeroIter].index = j;
            ret[nonzeroIter].value = description->at(i);
            nonzeroIter++;
        }
        ret[nonzeroIter].index = -1;//end
    }
    return ret;
}

vector<KeyPoint>* getKeyPoints(string option, const Mat& img)
{
    if (hasPrefix(option,"SIFT"))
    {
        Mat img;
        cvColor(color_img,img,CV_BGR2GRAY);
        
        int nfeaturePoints=10;
        int nOctivesPerLayer=3;
        SIFT detector(nfeaturePoints,nOctivesPerLayer);
        vector<KeyPoint>* ret = new vector<KeyPoint>();
        
        detector(img,noArray(),ret,noArray());
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
        cvColor(color_img,img,CV_BGR2GRAY);
        
        int nfeaturePoints=10;
        int nOctivesPerLayer=3;
        SIFT detector(nfeaturePoints,nOctivesPerLayer);
        Mat desc;
        detector(img,noArray(),*keyPoints,desc,true);
        vector< vector<double> >* ret = new vector< vector<double> >(desc.rows);
        for (int i=0; i<desc.rows; i++)
        {
            ret->at(i).resize(desc.cols);
            for (int j=0; j<desc.cols; j++)
            {
                ret->at(i).at(j) = desc.at<float>(i,j);
            }
        }
        return ret;
    }
    else if (hasPrefix(option,"customSIFT"))
    {
        Mat img;
        cvColor(color_img,img,CV_BGR2GRAY);
        
        vector< vector<double> >* ret = new vector< vector<double> >(desc.rows);
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
        
        for (const vector<double>& desc : descriptors)
        {
            vector< tuple<int,float> > quan = codebook->quantizeSoft(desc,LLC);
            for (const auto &v : quan)
            {
                ret->at(get<0>(v)) += get<1>(v);
            }
        }
        
        //Normalize the description
        double sum;
        for (double v : *ret)
            sum += v*v;
        double norm = sqrt(sum);
        for (double& v : *ret)
            v /= norm;
            
        return ret;
    }
    
    return NULL;
}
