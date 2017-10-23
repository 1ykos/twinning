/* Copyright (c) Wolfgang Brehm 2017, Greta Assmann 2016                     */
/* This is Alg2.1 for detwinning computationally twinned diffraction         */
/* All rights reserved                                                       */

// the asu library is needed, get it at https://github.com/1ykos/asu
#define BOOST_THREAD_PROVIDES_FUTURE

#include <array>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include <LBFGS.h>

using std::array;
using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;
using std::setw;
using std::swap;
using std::unordered_map;
using std::vector;

int main(int argc, char *argv[])
{
  vector<unordered_map<size_t,array<double,2>>> cctable(N);
  vector<size_t> assignments;
  size_t d = (argc>1)?(std::stoi(argv[1])):2;
  std::mt19937_64 mt(std::random_device{}());
  std::uniform_int_distribution<size_t> ud(0,d);
  while (cin){
    size_t i,j,n;
    double cc;
    cin >> i >> j >> cc >> n ;
    if (i>j) swap(i,j);
    cctable[i][j]={cc,n-3.0};
    cctable[j][i]={cc,n-3.0};
    assignments.push_back(ud(mt));
  }
  size_t s;
  double *scores = new double[d];
  double *sumwgt = new double[d];
  do{
    s=0;
    for (size_t i=0;i!=assignments.size();++i){
      cerr << i << endl;
      std::fill(scores,scores+d,0);
      std::fill(sumwgt,sumwgt+d,0);
      for (auto it=cctable[i].begin();it!=cctable[i].end();++it){
        scores[assignments[it->first]]+=it->second()[0]*it->second()[1];
        sumwgt[assignments[it->first]]+=it->second()[1];
      }
      size_t a = 0;
      double min=scores[0]/sumwgt[0];
      for (size_t j=1;j!=d;++j){
        if (scores[j]/sumwgt[j]<min){
          min = scores[j]/sumwgt[j];
          a = j;
        }
      }
      s+=(assignments[i]!=a);
      assignments[i]=a;
    }
  }while(s);
  delete[] scores;
  delete[] sumwgt;
  return 0;
}
