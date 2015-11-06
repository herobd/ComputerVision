#include "codebook.h"

#include <assert.h>
#include <sstream>
#include <iostream>

#define CONV_THRESH 0.001

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
            dist += (val*val);
        }
        dist = sqrt(dist);
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
    
    
    
    //Hack workaround. new
    for (int j=0; j<t; j++)
    {
        double quot = 0;
        for (int j_other=0; j_other<t; j_other++)
            quot += get<1>(q[j])/get<1>(q[j_other]);
        ret.push_back(make_tuple(get<0>(q[j]),1/quot));
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
            //stringstream s;
            //s << setprecision(15) << val;
            file << setprecision(15) << val << endl;
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
            double val = stof(line);
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


void Codebook::my_kmeans(const cv::Mat& data, int size, cv::Mat& centers)
{
    vector<float> maxs;
    maxs.resize(data.cols,0);
    vector<float> mins;
    mins.resize(data.cols,99999);
    for (int r=0; r<data.rows; r++)
    {
        for (int c=0; c<data.cols; c++)
        {
            if (data.at<float>(r,c) > maxs[c])
                maxs[c]=data.at<float>(r,c);
            if (data.at<float>(r,c) < mins[c])
                mins[c]=data.at<float>(r,c);
        }
    }
    
    bool cont = true;
    while(cont)
    {
        cont=false;
        centers = (cv::Mat_<float>(size,data.cols));
        default_random_engine generator;
        for (int r=0; r<centers.rows; r++)
        {
            for (int c=0; c<centers.cols; c++)
            {
                uniform_real_distribution<float> distribution(mins[c],maxs[c]);
                centers.at<float>(r,c) = distribution(generator);
            }
        }
        //vector<int> curClass(data.rows);
        
        for (int iter=0; iter<100; iter++)
        {
            cv::Mat sum(size,data.cols,CV_32F,cv::Scalar(0.));
            vector<int> count(size);
            for (int r=0; r<data.rows; r++)
            {
                int bestClass=0;
                int minDist=cv::norm(data.row(r) - centers.row(0));
                for (int cl=1; cl<size; cl++)
                {
                    float dist = cv::norm(data.row(r) - centers.row(cl));
                    if (dist < minDist)
                    {
                        bestClass=cl;
                        minDist=dist;
                    }
                }
                //curClass[r]=bestClass;
                sum.row(bestClass)=(sum.row(bestClass) + data.row(r));
                count[bestClass]++;
            }
            
            cout << "centers"<<endl;
            int conv=0;
            for (int cl=0; cl<size; cl++)
            {
                if (count[cl]==0)
                {
                    cont=true;
                    continue;
                }
                else if (count[cl]==1)
                {
                    cout << cl << " has 1"<<endl;
                }
                cv::Mat newVal = sum.row(cl)/count[cl];
                double dist = cv::norm(newVal - centers.row(cl));
                if (dist < CONV_THRESH)
                    conv++;
                //centers.row(cl) = (newVal+0);
                newVal.copyTo(centers.row(cl));
                
                for (int i=0; i<newVal.cols; i++)
                {
                    cout << centers.at<float>(cl,i) << ", ";
                    assert(centers.at<float>(cl,i) == newVal.at<float>(0,i));
                }
                cout << endl;
            }
            if (conv > size-1)
            {
                cout << "my_kmeans reached convergence" << endl;
                break;
            }
        }
    }
}


void Codebook::rescue(vector<int>& temp, cv::Mat& data, int codebook_size)
{
        assert(temp.size() == data.rows);
        vector< vector<float> > sums(codebook_size);
        vector<int> count(codebook_size);
        for (int r=0; r<temp.size(); r++)
        {
            int cl = temp[r];
            count.at(cl)++;
            if (sums[cl].size()==0)
                sums[cl].resize(data.cols,0);
            for (int c=0; c<data.cols; c++)
            {
                
                sums[cl][c] += data.at<float>(r,c);
            }
        }
        cout << "compiling codebook" << endl;
    
        for (int cl=0; cl<codebook_size; cl++)
        {
            assert(count[cl]>0);
            vector<double> toAdd;
            for (int c=0; c<data.cols; c++)
            {
                assert(sums[cl][c]>=0);
                toAdd.push_back(sums[cl][c]/count[cl]);
            }
            codebook.push_back(toAdd);
        }
        
}

void Codebook::trainFromExamples(int codebook_size,vector< vector<double> >& accum)
{
    cv::Mat centriods;
    cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,1000,.99);
    //      Mat data(accum.size(),accum[0].size(),CV_32F);
    //      for (int r=0; r< accum.size(); r++)
    //          for (int c=0; c<accum[0].size(); c++)
    //              data.at<float>(r,c) = accum[r][c];
    int examples_size = std::min( std::max(codebook_size*500, 100000), (int) accum.size());
    cv::Mat data(examples_size,accum[0].size(),CV_32F);
    
    cout << "selecting random set" << endl;
    cout << "really. accum is " << accum.size() << endl;
    for (int count=0; count< examples_size; count++)
    {
        int r=rand()%accum.size();
        int orig=r;
        while (accum[r].size()==0)
        {
            r = (1+r)%accum.size();
            if (r==orig)
            {
                cout << "ERROR: not enough descriptors" << endl;
                assert(false);
                exit(-1);
                //break;
            }
        }
        //if (r==orig) break;
        
        for (int c=0; c<data.cols; c++)
        {
            assert(accum[r][c] >= 0);
            data.at<float>(count,c) = accum[r][c];
        }
        accum[r].resize(0);
    }
    
    
    cout << "computing kmeans" << endl;
    
    vector<int> temp;
    cv::kmeans(data,codebook_size,temp,crit,7,cv::KMEANS_PP_CENTERS,centriods);
    
    //my_kmeans(data,codebook_size,centriods);
    if(centriods.rows != codebook_size)
    {
        cout << "Centriods is probably blank (not cb size), rescuing" << endl;
        //assert(temp.type() == cv::CV_64UI1);
        rescue(temp,data,codebook_size);
        return;
    }
    
    cout << "compiling codebook" << endl;
    
    for (int r=0; r<centriods.rows; r++)
    {
        vector<double> toAdd;
        for (int c=0; c<centriods.cols; c++)
        {
            if (centriods.at<float>(r,c) < -1)
            {
                cout << "Negative in centriods, rescuing" << endl;
                rescue(temp,data,codebook_size);
                return;
            }
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
    assert(score2>=score1);
    //assert(score1>=score0);
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
    //assert(score1>score0);
    assert(score0+score1+score2+score3>=.9999 && score0+score1+score2+score3<=1.0001);
    sumtest=0;
    for (const auto &v : quan)
    {
        sumtest+=get<1>(v);
    }
    assert(sumtest<1.0001 && sumtest>.9999);
    
    
    ////Test training of codebook////////
    codebook.clear();
    vector< vector<double> >accum(1000);
    default_random_engine generator;
    normal_distribution<double> distributionP(5.0,.10);
    normal_distribution<double> distributionN(2.0,.10);
    normal_distribution<double> distributionR(3.0,.10);
    for (int i=0; i<accum.size(); i++)
    {
        for (int j=0; j<5*2+2; j++)
        {
            if (i%5 == j%5 && j<10)
                accum[i].push_back(fabs(distributionP(generator)));
            else if (j<10)
                accum[i].push_back(fabs(distributionN(generator)));
            else
                accum[i].push_back(fabs(distributionR(generator)));
        }
        
    }
    for (int i=0; i<accum.size(); i++)
    {
        for (int j=0; j<accum[i].size(); j++)
        {
            assert(accum[i][j] != 0);
        }
    }
    
    trainFromExamples(5,accum);
    
    vector<bool> found(5);
    for (int i=0; i<size(); i++)
    {
        double hi=0;
        int index;
        double hi2=0;
        int index2;
        for (int j=0; j<codebook[i].size(); j++)
        {
            if (codebook[i][j] > hi)
            {
                hi2=hi;
                index2=index;
                hi=codebook[i][j];
                index=j;
            }
            else if (codebook[i][j] > hi2)
            {
                hi2=codebook[i][j];
                index2=j;
            }
        }
        assert (index%5 == index2%5);
        assert(!found[index%5]);
        found[index%5]=true;
    }
    for (int i=0; i<size(); i++)
        assert(found[i]);
    
    vector<float> term9 = {7,3,3.3,2,2,3,1.5,1.7,.6,1.5,.1,10};
    quan = quantizeSoft(term9,1);
    assert(quan.size()==1 && get<1>(quan[0])==1 && get<0>(quan[0])==cw_0);
    
    vector<float> term10 = {7,3,3.3,2,10,7.1,2,2,2,12,.1,10};
    quan = quantizeSoft(term10,3);
    score0=0;
    double score4=0;
    double scoreOther=0;
    for (auto p : quan)
    {
        if (get<0>(p) == cw_0)
            score0 = get<1>(p);
        else if (get<0>(p) == cw_4)
            score4 = get<1>(p);
        else
            scoreOther = get<1>(p);
    }
    assert(quan.size()==3 && score0!=0 && score4!=0 && scoreOther!=0);
    assert(score0>scoreOther);
    assert(score4>scoreOther);
    sumtest=0;
    for (const auto &v : quan)
    {
        sumtest+=get<1>(v);
    }
    assert(sumtest<1.0001 && sumtest>.9999);
    
    vector< vector<double> > copy(codebook.size());
    for (int i=0; i<codebook.size(); i++)
    {   
        for (int j=0; j<depth();   j++)
            copy[i].push_back(codebook[i][j]);
    }
    save("tmp.dat435");
    codebook.clear();
    inverseDocFreq.clear();
    readIn("tmp.dat435");
    for (int i=0; i<codebook.size(); i++)
    {   
        for (int j=0; j<depth();   j++)
            assert(fabs(copy[i][j] - codebook[i][j])<.00001);
    }
    //double check
    vector<bool> found_doublecheck(5);
    for (int i=0; i<size(); i++)
    {
        double hi=0;
        int index;
        double hi2=0;
        int index2;
        for (int j=0; j<codebook[i].size(); j++)
        {
            if (codebook[i][j] > hi)
            {
                hi2=hi;
                index2=index;
                hi=codebook[i][j];
                index=j;
            }
            else if (codebook[i][j] > hi2)
            {
                hi2=codebook[i][j];
                index2=j;
            }
        }
        assert (index%5 == index2%5);
        assert(!found_doublecheck[index%5]);
        found_doublecheck[index%5]=true;
    }
    for (int i=0; i<size(); i++)
        assert(found_doublecheck[i]);
        
        
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
