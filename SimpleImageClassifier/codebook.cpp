#include "codebook.h"

#include <assert.h>
#include <sstream>
#include <iostream>

bool Codebook::twentythree;

Codebook::Codebook()
{
}


void Codebook::push_back(vector<double> & exe)
{
    codebook.push_back(exe);
}

int Codebook::quantize(const vector<double> &term) const
{
    assert(term.size() == depth());
    int best=-1;
    double min=9999999999;
    for (int i=0; i<codebook.size(); i++)
    {
        double dist=0;
        for (int f=0; f<term.size(); f++)
        {
            dist += pow(term[f]-codebook[i][f],2);
        }
        
        if (dist<min)
        {
            min=dist;
            best=i;
        }
//        cout << "dist " << dist << endl;
    }
//    cout << "best " << best << ", min " << min << endl;
    return best;
}

int Codebook::quantize(const vector<float> &term) const
{
    assert(term.size() == depth());
    int best=-1;
    double min=9999999999;
    for (int i=0; i<codebook.size(); i++)
    {
        double dist=0;
        for (int f=0; f<term.size(); f++)
        {
            dist += pow(term[f]-codebook[i][f],2);
        }
        
        if (dist<min)
        {
            min=dist;
            best=i;
        }
//        cout << "dist " << dist << endl;
    }
//    cout << "best " << best << ", min " << min << endl;
    return best;
}

vector< tuple<int,float> > Codebook::quantizeSoft(const vector<double> &term, int t) const
{
    vector<float> temp;
    for (double n : term) temp.push_back((float)n);
    return quantizeSoft(temp,t);
}

vector< tuple<int,float> > Codebook::quantizeSoft(const vector<float> &term, int t) const
{
    
    vector< tuple<int,float> > ret; 
//    assert(term.size() == depth());
    if (term.size()==0)
        return ret;
    assert(term.size() == depth());
    
//    bool empty=true;
//    for (int f=0; f<term.size(); f++)
//    {
//        if (term[f] >0)
//        {
//            empty=false;
//            break;
//        }
//    }
//    if (empty) return ret;
    
    double minDist=numeric_limits<double>::max();
    
    vector< tuple<int,float> > q;
    int sizeC = codebook.size();
    int sizeT = term.size();
    for (int i=0; i<sizeC; i++)
    {
        double dist=0;
        for (int f=0; f<sizeT; f++)
        {
            double val =term[f]-codebook[i][f];
            dist += sqrt(val*val);
        }
        if (dist<minDist)
            minDist=dist;
//        cout << i<<": "<<dist<<endl;
        auto it = q.rbegin();
        while(1)
        {
            if (q.size() == 0)
            {
//                cout << "add first"<<endl;
                q.push_back(make_tuple(i,(float)dist));
                break;
            }
            else if (it==q.rend())
            {
//                cout << "add insert begining"<<endl;
                q.insert(it.base(),make_tuple(i,(float)dist));
                break;
            }
            else if (dist>=get<1>(*it))
            {
                if (it == q.rbegin() && q.size()<=t)
                {
//                    cout << "add end, 'cause "<<dist<<">="<<get<1>(*it)<<endl;
                    q.push_back(make_tuple(i,(float)dist));
                }
                else
                {
//                    cout << "add insert mid, 'cause "<<dist<<">="<<get<1>(*it)<<endl;
                    q.insert(it.base(),make_tuple(i,(float)dist));
                }
                break;
            }
            
            
            it++;
        }
        if (q.size() > t)
            q.pop_back();
//        cout << "dist " << dist << endl;
    }
    
    if (t==1)
    {
        ret.push_back(make_tuple(get<0>(q.front()),1));
        return ret;
    }
    
    //Set up and perform a least squares solve
//    cv::Mat X(term.size(),t,CV_32F);
//    for (int i=0; i<term.size(); i++)
//        for (int j=0; j<t; j++)
//        {
//            X.at<float>(i,j)=codebook[get<0>(q[j])][i];
//        }
//    cv::Mat y(term.size(),1, CV_32F);
//    for (int i=0; i<term.size(); i++)
//    {
//        y.at<float>(i,0) = term[i];
//    }
    
//    cv::Mat inter = (X.t()*X).inv(cv::DECOMP_CHOLESKY);
//    cv::Mat B = inter * X.t() * y;
//    cout <<"X:\n "<<X <<"\ninter:\n " << inter <<"\nB:\n " << B << endl;
    
//    for (int j=0; j<t; j++)
//    {
//        ret.push_back(make_tuple(get<0>(q[j]),B.at<float>(j,0)));
//    }
    
//    //From "The Devil is in the Details"
//    cv::Mat delta_i(term.size(),t,CV_64F);
//    for (int i=0; i<term.size(); i++)
//        for (int j=0; j<t; j++)
//        {
//            delta_i.at<double>(i,j) = (double) (term[i]-codebook[get<0>(q[j])][i]);///sqrt(minDist/2);
//        }
    
//    double reg = 0.01;
//    cv::Mat inter = (delta_i.t()*delta_i + reg*cv::Mat::eye(t,t,CV_64F)).inv();
////    cout << "sqr:\n "<<delta_i.t()*delta_i<<"\ninter:\n "<<inter<<endl;
//    cv::Mat a = inter * delta_i.t()* cv::Mat::ones(term.size(),1,CV_64F);
    

////    double min=0;
////    for (int ii=0; ii<t; ii++)
////    {
////        if(a.at<double>(ii,0)<min)
////        {
////            min=a.at<double>(ii,0);
////        }
////    }
////    if (min<0)
////    {
////        for (int ii=0; ii<t; ii++)
////        {
////            a.at<double>(ii,0) -= min;
////        }
////    }
    
//    //    bool neg = a.at<double>(0,0)<0;
////    for (int ii=0; ii<t; ii++)
////        if(a.at<double>(ii,0)<0 && !neg)
////        {
////            a.at<double>(ii,0)=0;
////        }
////        else if(a.at<double>(ii,0)>0 && neg)
////        {
////            for (int iii=0; iii<ii; iii++)
////                a.at<double>(iii,0)=0;
////            neg=false;
////        }
    
//    double norm = cv::norm(cv::Mat::ones(1,t,CV_64F)*a);
//    //if (norm==0) norm=1;
//    for (int j=0; j<t; j++)
//    {
//        ret.push_back(make_tuple(get<0>(q[j]),(a.at<double>(j,0)/norm)));
//    }

    
    
//    //from http://cyberzhg.github.io/blog/Computer-Vision/Locality-Constrained-Linear-Coding/
//    cv::Mat B(t,term.size(),CV_64F);
//    cv::Mat x(1,term.size(),CV_64F);
//    for (int i=0; i<term.size(); i++)
//    {
//        x.at<double>(0,i)=term[i];
//        for (int j=0; j<t; j++)
//        {
//            B.at<double>(j,i) = codebook[get<0>(q[j])][i];
//        }
//    }
    
    
//    cv::Mat z = B-cv::repeat(x,t,1);
//    cv::Mat C = z*z.t();
//    double trace=cv::trace(C)[0];
//    C = (C + (0.0001*trace*cv::Mat::eye(t,t,CV_64F))).inv() * cv::Mat::ones(t,1,CV_64F);
//    ///Ignore negative values? Improves simple by 5 (w/o power norm)
//    for (int j=0; j<t; j++)
//    {
//        if (C.at<double>(j,0)<0)
//            C.at<double>(j,0)=0;
//    }
//    //////////////////////////
//    double norm = cv::norm(cv::Mat::ones(1,t,CV_64F)*C);
//    for (int j=0; j<t; j++)
//    {
//        ret.push_back(make_tuple(get<0>(q[j]),(C.at<double>(j,0)/norm)));
//    }
    
    
    
    //Hack workaround. Improves above by 10 on simple
    float total=0;
    for (int j=0; j<t; j++)
    {
        total += get<1>(q[j]);
    }
    for (int j=0; j<t; j++)
    {
        ret.push_back(make_tuple(get<0>(q[j]),get<1>(q[j])/total));
    }
    
    
//    cout << "best " << best << ", min " << min << endl;
    double sumtest=0;
//    int counttest=0;
    for (const auto &v : ret)
    {
        sumtest+=get<1>(v);
        //assert (fabs(get<1>(v))<6);
//        if (get<1>(v)<0)
//        {
//            //cout << "neg val " << get<1>(v) << endl;
//            counttest++;
//        }
    }
//    if (counttest == t)
//    {
//        cout << "all neg" << endl;
//    }
//    for (int ii=0; ii<t; ii++)
//        assert(C.at<double>(ii,0)>=0);
    assert(sumtest<1.0001 && sumtest>.9999);
    return ret;
}

int Codebook::quantize(const cv::Mat &term) const
{
    assert(term.cols == depth());
    int best=-1;
    double min=9999999999;
    for (int i=0; i<codebook.size(); i++)
    {
        double dist=0;
        for (int f=0; f<term.cols; f++)
        {
            dist += pow(term.at<float>(0,f)-codebook[i][f],2);
        }
        
        if (dist<min)
        {
            min=dist;
            best=i;
        }
//        cout << "dist " << dist << endl;
    }
//    cout << "best " << best << ", min " << min << endl;
    return best;
}




void Codebook::save(string filePath)
{
    ofstream file;
    file.open (filePath, ios::out);
    
    file << codebook.size() << endl;
    file << codebook[0].size() << endl;
    for (vector<double> exemplar : codebook)
    {
        for (double val : exemplar)
        {
            file << val << endl;
        }
    }
    file.close();
}

void Codebook::readIn(string filePath)
{
    ifstream file;
    file.open (filePath, ios::in);
    string line;
    getline (file,line);
    int numOfEntries = stoi(line);
//    assert(numOfEntries==CODEBOOK_SIZE);
    getline (file,line);
    int numOfFeatures = stoi(line);
    for (int n=0; n < numOfEntries; n++)
    {
        vector<double> exemplar;
        for (int f=0; f < numOfFeatures; f++)
        {
            getline (file,line);
            int val = stoi(line);
            exemplar.push_back(val);
        }
        push_back(exemplar);
    }
    file.close();
}
void Codebook::readInCSV(string filePath)
{
    ifstream file;
    file.open (filePath, ios::in);
    string line;
//    getline (file,line);
//    int numOfEntries = stoi(line);
//    assert(numOfEntries==CODEBOOK_SIZE);
//    getline (file,line);
//    int numOfFeatures = stoi(line);
//    for (int n=0; n < numOfEntries; n++)
//    {
//        vector<double> exemplar;
//        for (int f=0; f < numOfFeatures; f++)
//        {
//            getline (file,line);
//            int val = stoi(line);
//            exemplar.push_back(val);
//        }
//        push_back(exemplar);
//    }
    
    while (getline(file,line))
    {
        vector<double> exemplar;
        
        istringstream ss(line);
        string token;
        
        while(getline(ss, token, ',')) {
            exemplar.push_back( stod(token) );
        }
        double freq = exemplar.back();
        exemplar.pop_back();
        inverseDocFreq.push_back(freq);
        codebook.push_back(exemplar);
    }
    
    
    
    file.close();
}

void Codebook::trainFromExamples(int codebook_size,vector< vector<float> >& accum)
{
    Mat centriods;
    TermCriteria crit(TermCriteria::COUNT + TermCriteria::EPS,500,.9);
    //      Mat data(accum.size(),accum[0].size(),CV_32F);
    //      for (int r=0; r< accum.size(); r++)
    //          for (int c=0; c<accum[0].size(); c++)
    //              data.at<float>(r,c) = accum[r][c];
    Mat data(codebook_size*300,accum[0].size(),CV_32F);
    
    cout << "selecting random set" << endl;
    cout << "really. accum is " << accum.size() << endl;
    for (int count=0; count< codebook_size*300; count++)
    {
        int r=rand()%accum.size();
        int orig=r;
        while (accum[r].size()==0)
        {
            r = (1+r)%accum.size();
            if (r==orig)
            {
                cout << "ERROR: not enough descriptors" << endl;
                return NULL;
            }
        }
        
        
        for (int c=0; c<accum[0].size(); c++)
        {
            assert(accum[r][c] >= 0);
            data.at<float>(count,c) = accum[r][c];
        }
        accum[r].resize(0);
    }
    cout << "computing kmeans" << endl;
    
    Mat temp;
    //      Kmeans(data,codebook_size,temp,crit,10,KMEANS_RANDOM_CENTERS,&centriods);
    kmeans(data,codebook_size,temp,crit,10,KMEANS_RANDOM_CENTERS,centriods);
    
    
    cout << "compiling codebook" << endl;
    
    for (int r=0; r<centriods.rows; r++)
    {
        vector<double> toAdd;
        for (int c=0; c<centriods.cols; c++)
        {
            assert(centriods.at<float>(r,c) >= 0);
            toAdd.push_back(centriods.at<float>(r,c));
        }
        codebook.push_back(toAdd);
    }
  
  
}

void Codebook::unittest()
{
    ofstream file;
    file.open ("tmp.dat435", ios::out);
    
    file << 3 << endl;
    file << 3 << endl;
    
    file << 1 << endl;
    file << 0 << endl;
    file << 0 << endl;
    
    file << 0 << endl;
    file << 1 << endl;
    file << 0 << endl;
    
    file << 0 << endl;
    file << 0 << endl;
    file << 1 << endl;
    
    file.close();
    readIn("tmp.dat435");
    save("tmp.dat435");
    codebook.clear();
    inverseDocFreq.clear();
    readIn("tmp.dat435");
    
    vector<float> term = {.5,0,.1};
    vector< tuple<int,float> > quan = quantizeSoft(term,1);
    assert(get<0>(quan[0])==0 && get<1>(quan[0])==1);
    
    vector<float> term2 = {.3,0,.3};
    quan = quantizeSoft(term2,2);
    assert(get<1>(quan[0])==.5 && get<1>(quan[1])==.5);
    
    file.open ("tmp.dat435", ios::out);
    
    file << 4 << endl;
    file << 4 << endl;
    
    file << 1 << endl;
    file << 0 << endl;
    file << 0 << endl;
    file << 0 << endl;
    
    file << 0 << endl;
    file << 1 << endl;
    file << 0 << endl;
    file << 0 << endl;
    
    file << 0 << endl;
    file << 0 << endl;
    file << 1 << endl;
    file << 0 << endl;
    
    file << 0 << endl;
    file << 0 << endl;
    file << 0 << endl;
    file << 1 << endl;
    
    file.close();
    codebook.clear();
    inverseDocFreq.clear();
    readIn("tmp.dat435");
    
    vector<float> term3 = {.3,0,.2,.1};
    quan = quantizeSoft(term3,3);
    double score0=0;
    double score1=0;
    double score2=0;
    double score3=0;
    for (auto p : quan)
    {
        if (get<0>(p) == 0)
            score0 = get<1>(p);
        else if (get<0>(p) == 1)
            score1 = get<1>(p);
        else if (get<0>(p) == 2)
            score2 = get<1>(p);
        else if (get<0>(p) == 3)
            score3 = get<1>(p);
    }
    assert(score0>score2);
    assert(score2>score3);
    assert(score3>score1);
    assert(score0+score1+score2+score3>=.9999 && score0+score1+score2+score3<=1.0001);
    double sumtest=0;
    for (const auto &v : quan)
    {
        sumtest+=get<1>(v);
    }
    assert(sumtest<1.0001 && sumtest>.9999);
    
    vector<float> term4 = {.1,.05,.3,.2};
    quan = quantizeSoft(term4,4);
    score0=0;
    score1=0;
    score2=0;
    score3=0;
    for (auto p : quan)
    {
        if (get<0>(p) == 0)
            score0 = get<1>(p);
        else if (get<0>(p) == 1)
            score1 = get<1>(p);
        else if (get<0>(p) == 2)
            score2 = get<1>(p);
        else if (get<0>(p) == 3)
            score3 = get<1>(p);
    }
    assert(score2>score3);
    assert(score3>score0);
    assert(score0>score1);
    assert(score0+score1+score2+score3>=.9999 && score0+score1+score2+score3<=1.0001);
    sumtest=0;
    for (const auto &v : quan)
    {
        sumtest+=get<1>(v);
    }
    assert(sumtest<1.0001 && sumtest>.9999);
    
    vector<float> term5 = {.1,.05,.3,.1};
    quan = quantizeSoft(term5,3);
    score0=0;
    score1=0;
    score2=0;
    score3=0;
    for (auto p : quan)
    {
        if (get<0>(p) == 0)
            score0 = get<1>(p);
        else if (get<0>(p) == 1)
            score1 = get<1>(p);
        else if (get<0>(p) == 2)
            score2 = get<1>(p);
        else if (get<0>(p) == 3)
            score3 = get<1>(p);
    }
    assert(score2>score3);
    assert(score3==score0);
    assert(score0>score1);
    assert(score0+score1+score2+score3>=.9999 && score0+score1+score2+score3<=1.0001);
    sumtest=0;
    for (const auto &v : quan)
    {
        sumtest+=get<1>(v);
    }
    assert(sumtest<1.0001 && sumtest>.9999);
    
    vector<float> term6 = {.05,.1,.2,.3};
    quan = quantizeSoft(term6,4);
    score0=0;
    score1=0;
    score2=0;
    score3=0;
    for (auto p : quan)
    {
        if (get<0>(p) == 0)
            score0 = get<1>(p);
        else if (get<0>(p) == 1)
            score1 = get<1>(p);
        else if (get<0>(p) == 2)
            score2 = get<1>(p);
        else if (get<0>(p) == 3)
            score3 = get<1>(p);
    }
    assert(score3>score2);
    assert(score2>score1);
    assert(score1>score0);
    assert(score0+score1+score2+score3>=.9999 && score0+score1+score2+score3<=1.0001);
    sumtest=0;
    for (const auto &v : quan)
    {
        sumtest+=get<1>(v);
    }
    assert(sumtest<1.0001 && sumtest>.9999);
    
    vector<float> term7 = {.05,1,100,5000};
    quan = quantizeSoft(term7,4);
    score0=0;
    score1=0;
    score2=0;
    score3=0;
    for (auto p : quan)
    {
        if (get<0>(p) == 0)
            score0 = get<1>(p);
        else if (get<0>(p) == 1)
            score1 = get<1>(p);
        else if (get<0>(p) == 2)
            score2 = get<1>(p);
        else if (get<0>(p) == 3)
            score3 = get<1>(p);
    }
    assert(score3>score2);
    assert(score2>score1);
    assert(score1>score0);
    assert(score0+score1+score2+score3>=.9999 && score0+score1+score2+score3<=1.0001);
    sumtest=0;
    for (const auto &v : quan)
    {
        sumtest+=get<1>(v);
    }
    assert(sumtest<1.0001 && sumtest>.9999);
    
    vector<float> term8 = {.05,1,1000,5000};
    quan = quantizeSoft(term8,4);
    score0=0;
    score1=0;
    score2=0;
    score3=0;
    for (auto p : quan)
    {
        if (get<0>(p) == 0)
            score0 = get<1>(p);
        else if (get<0>(p) == 1)
            score1 = get<1>(p);
        else if (get<0>(p) == 2)
            score2 = get<1>(p);
        else if (get<0>(p) == 3)
            score3 = get<1>(p);
    }
    assert(score3>score2);
    assert(score2>score1);
    assert(score1>score0);
    assert(score0+score1+score2+score3>=.9999 && score0+score1+score2+score3<=1.0001);
    sumtest=0;
    for (const auto &v : quan)
    {
        sumtest+=get<1>(v);
    }
    assert(sumtest<1.0001 && sumtest>.9999);
    
    
//    codebook.clear();
//    for (double y=0.1; y<6; y+=2+.05*(int)y)
//        for (double x=0.1; x<5; x+=2+1.33*(int)y)
//            for (double z=0.1; z<5; z+=2)
//        {
//            vector<double> v= {x,y,z};
//            push_back(v);
//        }
//    vector<double> t1= {0.1,.4,.2};
//    int t1q=quantize(t1);
////    assert(t1q==0);
    
//    vector<double> t2= {1,2,1};
//    vector< tuple<int,float> > t2q = quantizeSoft(t2,2);  
////    assert(fabs(get<1>(t2q[0])-.5)<.001 && fabs(get<1>(t2q[1])-.5)<.001);
    
//    vector<double> t3= {1,3,4};
//    vector< tuple<int,float> > t3q = quantizeSoft(t3,4);  
////    assert(fabs(get<1>(t3q[0])-.333)<.001 && fabs(get<1>(t3q[1])-.333)<.001 && fabs(get<1>(t3q[2])-.333)<.001);
    cout << "Codebook passed its tests!" << endl;
}

void Codebook::print()
{
    for (int i=0; i<codebook.size(); i++)
    {
        cout << i << "::\t";
        for (float f : codebook[i])
            cout << f << ",\t";
        cout << endl;
    }
}
