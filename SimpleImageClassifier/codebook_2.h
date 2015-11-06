#ifndef CODEBOOK_2_H
#define CODEBOOK_2_H
#define CODEBOOK_H

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <math.h>
#include "opencv2/core/core.hpp"
#include <assert.h>
#include <limits>
#include <iomanip>
using namespace std;

#define CODEBOOK_SIZE /*4096*/100

class Codebook
{
public:
    Codebook();
    vector< tuple<int,float> > quantizeSoft(const vector<double> &term, int t) const;
    vector< tuple<int,float> > quantizeSoft(const vector<float> &term, int t) const;
    int quantize(const vector<double> &term) const {return get<0>(quantizeSoft(term,1)[0]);}
    int quantize(const vector<float> &term) const {return get<0>(quantizeSoft(term,1)[0]);}
    int quantize(const cv::Mat &term) const {cout << "ERROR: Codebook:quantize(Mat) not implemented" << endl;exit(-1);return -1;}
    unsigned int size() const {return codebook.size();}
    unsigned int depth() const {if (codebook.size()>1) return codebook[0].size(); return 0;}
    double getInverseDocFreq(int i) const {assert(i<codebook.size()); return inverseDocFreq[i];}
    void trainFromExamples(int codebook_size,vector< vector<float> >& accum);
    
    void save(string filePath);
    void readIn(string filePath);
    void readInCSV(string filePath);
    
    void unittest();
    void print();
    static double my_stof(string s)
    {
       istringstream os(s);
       double d;
       os >> d;
       return d;
    }
    
    vector<double> getEx(int ex){return codebook[ex];}
    void push_back(vector<double>& v) {codebook.push_back(v);}
private:
    vector< vector<double> > codebook;
    vector<double> inverseDocFreq;
    
    double distance(const vector<float> &one, const vector<double> &two) const;
};

#endif // CODEBOOK_H

