#ifndef CODEBOOK_H
#define CODEBOOK_H

#include <fstream>
#include <vector>
#include <queue>
#include <math.h>
#include "opencv2/core/core.hpp"
#include <assert.h>
#include <limits>
using namespace std;

#define CODEBOOK_SIZE /*4096*/100

class Codebook
{
public:
    Codebook();
    int quantize(const vector<double> &term) const;
    int quantize(const vector<float> &term) const;
    int quantize(const cv::Mat &term) const;
    vector< tuple<int,float> > quantizeSoft(const vector<double> &term, int t) const;
    vector< tuple<int,float> > quantizeSoft(const vector<float> &term, int t) const;
    void push_back(vector<double> & exe);
    unsigned int size() const {return codebook.size();}
    unsigned int depth() const {if (codebook.size()>1) return codebook[0].size(); return 0;}
    double getInverseDocFreq(int i) const {assert(i<codebook.size()); return inverseDocFreq[i];}
    void save(string filePath);
    void readIn(string filePath);
    void readInCSV(string filePath);
    
    void trainFromExamples(int codebook_size,vector< vector<float> >& accum);
    
    void unittest();
    void print();
    static bool twentythree;
    
    vector<double> getEx(int ex){return codebook[ex];}
    
private:
    vector< vector<double> > codebook;
    vector<double> inverseDocFreq;
};

#endif // CODEBOOK_H
