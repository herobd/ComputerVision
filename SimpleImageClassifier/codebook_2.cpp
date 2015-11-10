
#include "codebook_2.h"

Codebook::Codebook(){}

/*int Codebook::quantize(const vector<double> &term) const
{
    vector<float> temp(term.begin(), term.end();
    return quantize(temp);
}

int Codebook::quantize(const vector<float> &term) const
{
    return -1;
}*/

double Codebook::distance(const vector<float> &one, const vector<double> &two) const
{
    double dist=0;
    assert(one.size()==two.size());
    for (int i=0; i<one.size(); i++)
    {
        dist += pow(one[i]-two[i],2);
    }
    return sqrt(dist);
}

vector< tuple<int,float> > Codebook::quantizeSoft(const vector<double> &term, int t) const
{
    vector<float> temp(term.begin(), term.end());
    return quantizeSoft(temp,t);
}

vector< tuple<int,float> > Codebook::quantizeSoft(const vector<float> &term, int t) const
{
    vector< tuple<int,double> > top_cwDist(1);
    top_cwDist[0] = make_tuple(0,distance(term,codebook[0]));
    for (int cw=1; cw<t; cw++)
    {
        double d = distance(term,codebook[cw]);
        auto iter = top_cwDist.begin();
        while (iter!=top_cwDist.end() && get<1>(*iter)<=d)
            iter++;
        if (iter!=top_cwDist.end())
        {
            top_cwDist.insert(iter, make_tuple(cw,d));
        }
        else
        {
            top_cwDist.push_back(make_tuple(cw,d));
        }
    }
    for (int cw=t; cw<size(); cw++)
    {
        double d = distance(term,codebook[cw]);
        auto iter = top_cwDist.begin();
        while (iter!=top_cwDist.end() && get<1>(*iter)<=d)
            iter++;
        if (iter!=top_cwDist.end())
        {
            top_cwDist.insert(iter, make_tuple(cw,d));
            if (top_cwDist.size()>t)
                top_cwDist.pop_back();
        }
    }
    
    vector< tuple<int,float> > ret;
    
    for (int j=0; j<t; j++)
    {
        double quot = 0;
        for (int j_other=0; j_other<t; j_other++)
            quot += get<1>(top_cwDist[j])/get<1>(top_cwDist[j_other]);
        ret.push_back(make_tuple(get<0>(top_cwDist[j]),1/quot));
    }
    return ret;
}

void Codebook::trainFromExamples(int codebook_size,vector< vector<double> >& accum)
{
    int dataSize = min((int)accum.size(), (int)700*codebook_size);
    cv::Mat data(dataSize,accum[0].size(), CV_32F);
    vector<bool> used(accum.size());
    for (int row=0; row<dataSize; row++)
    {
        int r = rand()%accum.size();
	int orig = r;
	while (used[r])
	{
	    r = (r+1)%accum.size();
        if (r == orig)
	    {
		    cout << "ERROR, not enough data for kmeans!" << endl;
		    assert(false);
		    exit(-1);
	    }
	}
	for (int col=0; col<data.cols; col++)
	{
            data.at<float>(row,col) = accum[r][col];
	}
    }
    
    vector<int> labels;
    cv::Mat centriods;
    cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,1000,.99);
    cv::kmeans(data,codebook_size,labels,crit,7,cv::KMEANS_PP_CENTERS,centriods);
    
    assert(centriods.rows == codebook_size);
    
    for (int cw=0; cw<codebook_size; cw++)
    {
	    vector<double> codeword(centriods.cols);
	    for (int col=0; col<centriods.cols; col++)
	    {
		    codeword[col] = centriods.at<float>(cw,col);
	    }
	    codebook.push_back(codeword);
    }

}

void Codebook::save(string filePath)
{
	ofstream outFile(filePath);
	if (!outFile)
	{
	    cout << "Error, could not open codebook save file: " << filePath << endl;
	    exit(-1);
	}
	outFile << size() << endl;
	outFile << depth() << endl;
	for (int cw=0; cw<size(); cw++)
	{
	    for (int d=0; d<depth(); d++)
	        outFile << setprecision(15) << codebook[cw][d] << endl;
	}
	outFile.close();
}


void Codebook::readIn(string filePath)
{
    ifstream in(filePath);
    if (!in)
	{
	    cout << "Error, could not open codebook save file: " << filePath << endl;
	    exit(-1);
	}
	string line;
	getline(in,line);
	int size = stoi(line);
	getline(in,line);
	int depth = stoi(line);
	for (int cw=0; cw<size; cw++)
	{
	    vector<double> codeword(depth);
	    for (int d=0; d<depth; d++)
	    {
	        getline(in,line);
	        codeword[d] = my_stof(line);
        }
        codebook.push_back(codeword);
	}
}




/////////////////
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
    
    int cw_0;
    int cw_4;
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
        if (index%5==0)
            cw_0=i;
        if (index%5==4)
            cw_4=i;
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

