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

#include "svm.h"

#include "customsift.h"
#include "codebook.h"

#define SAVE_LOC string("./save/")
#define DEFAULT_IMG_DIR string("/home/brian/Douments/CS601R/leedsbutterfly/images/")
#define DEFAULT_CODEBOOK_SIZE 4096

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
        }
        ret[nonzeroIter].index = -1;//end
    }
    return ret;
}

vector<KeyPoint>* getKeyPoints(string option, const Mat& color_img)
{
    if (hasPrefix(option,"SIFT"))
    {
        Mat img;
        cvtColor(color_img,img,CV_BGR2GRAY);
        
        int nfeaturePoints=10;
        int nOctivesPerLayer=3;
        SIFT detector(nfeaturePoints,nOctivesPerLayer);
        vector<KeyPoint>* ret = new vector<KeyPoint>();
        
        detector(img,noArray(),*ret,noArray(),false);
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
        
        int nfeaturePoints=10;
        int nOctivesPerLayer=3;
        SIFT detector(nfeaturePoints,nOctivesPerLayer);
        Mat desc;
        detector(img,noArray(),*keyPoints,desc,true);
        vector< vector<double> >* ret = new vector< vector<double> >(desc.rows);
        for (unsigned int i=0; i<desc.rows; i++)
        {
            ret->at(i).resize(desc.cols);
            for (unsigned int j=0; j<desc.cols; j++)
            {
                ret->at(i).at(j) = desc.at<float>(i,j);
            }
        }
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


void getImageDescriptions(string imageDir, bool trainImages, int positiveClass, Codebook* codebook, string keyPoint_option, string descriptor_option, string pool_option, vector< vector<double>* >* imageDescriptions, vector<double>* imageLabels)
{
    string saveFileName=SAVE_LOC+"imageDesc_"+(positiveClass!=-1?"one_":"")+keyPoint_option+"_"+descriptor_option+"_"+pool_option+".save";
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
                if (positiveClass!=-1)
                    imageLabels->push_back(stoi(sm[1])==positiveClass?1:-1);
                else
                    imageLabels->push_back(stoi(sm[1]));
              }
          }
        }
        
        int loopCrit = (int)fileNames.size();
        #pragma omp parallel for 
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
        
        //TODO save
        ofstream save(saveFileName);
        for (unsigned int i=0; i<imageDescriptions->size(); i++)
        {
            save << "Label:" << imageLabels->at(i) << " Desc:";
            for (unsigned int j=0; j<imageDescriptions->at(i)->size(); j++)
            {
                save << imageDescriptions->at(i)->at(j) << ",";
            }
            save << endl;
        }
        save.close();
    }
    else
    {
        string line;
        while (getline(load,line))
        {
            smatch sm;
            regex parse_option("Label:([0-9]+) Desc:(-?[0-9]*(\\.[0-9]+)?,)+");
            if(regex_search(line,sm,parse_option))
            {
                int label = stoi(sm[1]);
                if (positiveClass!=-1)
                    imageLabels->push_back(label==positiveClass?1:-1);
                else
                    imageLabels->push_back(label);
                vector<double>* imageDescription = new vector<double>();
                for (unsigned int i=2; i<sm.size(); i+=2)
                {
                    imageDescription->push_back(stof(sm[i]));
                }
            }
        }
        load.close();
    }
}
///////////////////////////////////////////

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
          //private(fileName,img,desc,t)
          int loopCrit = min((int)5000,(int)fileNames.size());
    #pragma omp parallel for 
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
                      //              assert(get<0>(t).size() > 0);
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
        
        string keypoint_option = argv[2];
        string descriptor_option = argv[3];
        string pool_option = argv[4];
        
        string imageDir = argv[argc-3];
        if (imageDir == "default") imageDir = DEFAULT_IMG_DIR;
        if (imageDir[imageDir.size()-1]!='/') imageDir += '/';
        
        string codebookLoc = argv[argc-2];
        if (codebookLoc == "default") codebookLoc = SAVE_LOC+"codebook_"+to_string(DEFAULT_CODEBOOK_SIZE)+"_"+keypoint_option+"_"+descriptor_option+".cb";
        string modelLoc = argv[argc-1];
        if (modelLoc == "default") modelLoc = SAVE_LOC+"model_"+to_string(positiveClass)+"_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+".svm";
        
        Codebook codebook;
        codebook.readIn(codebookLoc);
        
        vector< vector<double>* > imageDescriptions;
        vector<double> imageLabels;
        
        
        getImageDescriptions(imageDir, true, positiveClass, &codebook, keypoint_option, descriptor_option, pool_option, &imageDescriptions, &imageLabels);
        
        
        
        
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
        para->svm_type = C_SVC;
        para->kernel_type = RBF;
        para->gamma = 1.0/codebook.size();
        
        para->cache_size = 100;
        para->eps = eps;
        para->C = C;
        para->nr_weight=0;
        
        const char *err = svm_check_parameter(prob,para);
        if (err!=NULL)
        {
            cout << "ERROR: " << string(err) << endl;
            return -1;
        }
        struct svm_model *trained_model = svm_train(prob,para);
        
        int err2 = svm_save_model(modelLoc.c_str(), trained_model);
        
        if (err2 == -1)
        {
            cout << "ERROR: failed to save model" << endl;
            return -1;
        }
        
        int numLabels = svm_get_nr_class(trained_model);
        double* dec_values = new double[numLabels*(numLabels-1)/2];
        
        int correct = 0;
        for (unsigned int i=0; i<imageDescriptions.size(); i++)
        {
            struct svm_node* x = prob->x[i];
            double class_prediction = svm_predict_values(trained_model, x, dec_values);
            if (class_prediction == imageLabels[i])
                correct++;
        }
        cout << "Accuracy on training data: " << correct/(double)imageDescriptions.size() << endl;
        
        svm_free_model_content(trained_model);
        svm_destroy_param(para);
        delete[] dec_values;
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
        
        string keypoint_option = argv[2];
        string descriptor_option = argv[3];
        string pool_option = argv[4];
        
        string imageDir = argv[argc-3];
        if (imageDir == "default") imageDir = DEFAULT_IMG_DIR;
        if (imageDir[imageDir.size()-1]!='/') imageDir += '/';
        string codebookLoc = argv[argc-2];
        if (codebookLoc == "default") codebookLoc = SAVE_LOC+"codebook_"+to_string(DEFAULT_CODEBOOK_SIZE)+"_"+keypoint_option+"_"+descriptor_option+".cb";
        string modelLoc = argv[argc-1];
        if (modelLoc == "default") modelLoc = SAVE_LOC+"model_"+to_string(positiveClass)+"_"+keypoint_option+"_"+descriptor_option+"_"+pool_option+".svm";
        
        struct svm_model* trained_model = svm_load_model(modelLoc.c_str());
        
        
        Codebook codebook;
        codebook.readIn(codebookLoc);
        vector< vector<double>* > imageDescriptions;
        vector<double> imageLabels;
        
        
        getImageDescriptions(imageDir, true, positiveClass, &codebook, keypoint_option, descriptor_option, pool_option, &imageDescriptions, &imageLabels);
        
        int numLabels = svm_get_nr_class(trained_model);
        int* labels = new int[numLabels];
        svm_get_labels(trained_model, labels);
        double* dec_values = new double[numLabels*(numLabels-1)/2];
        
        
        //if (positiveClass==-1)
        //    for (unsigned int classIdx=0; classIdx<numLabels; classIdx++)
        //    {
                int correct = 0;
                for (unsigned int i=0; i<imageDescriptions.size(); i++)
                {
                    struct svm_node* x = convertDescription(imageDescriptions[i]);
                    double class_prediction = svm_predict_values(trained_model, x, dec_values);
                    if (class_prediction == imageLabels[i])
                        correct++;
                    delete imageDescriptions[i];
                }
                cout << "Accuracy: " << correct/(double)imageDescriptions.size() << endl;
        //    }
        //else
        //    cout << "Warning, one class not implemented" << endl;
        svm_free_model_content(trained_model);
        delete[] labels;
    }
    
    return 0;
}



