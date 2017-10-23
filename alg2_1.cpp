/* Copyright (c) Wolfgang Brehm 2017, Greta Assmann 2016                     */
/* This is Alg2.1 for detwinning computationally twinned diffraction         */
/* All rights reserved                                                       */

// the asu library is needed, get it at https://github.com/1ykos/asu
#define BOOST_THREAD_PROVIDES_FUTURE

#include <array>
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
using std::atanh;

const size_t D = 3;

double constexpr atanh_taylor13(
    const double a){
  double r = a;
  const double a2=a*a;
  double b = a;
  for (size_t i=1;i!=7;++i){
    b*=a2;
    r+=b/(i*2+1);
  }
  return r;
}

double constexpr diff_atanh_taylor13(
    const double a){
  const double a2=a*a;
  double b = a2;
  double r = 1+b;
  for (size_t i=2;i!=7;++i){
    b*=a2;
    r+=b;
  }
  return r;
}

struct alg2_1_target{
  const vector<unordered_map<size_t,array<double,2>>> cctable;
  double const operator()(
      const Eigen::VectorXd& x,
      Eigen::VectorXd& grad
      ) const {
    double s = 0;
    grad = Eigen::VectorXd::Zero(D*cctable.size());
    for (size_t i=1;i!=cctable.size();++i){
      for (size_t j=0;j!=i;++j){
        if (!cctable[i].count(j)) continue;
        const double w = cctable[i].at(j)[1];
        //const double atanh_cc = atanh(cctable[i].at(j)[0]);
        const double atanh_cc = atanh_taylor13(cctable[i].at(j)[0]);
        double dt = 0;
        for (size_t k=0;k!=D;++k) dt+=x(D*i+k)*x(D*j+k); //dotproduct
        //const double atanh_dt = atanh(dt); // atanh_taylor13(dt);
        const double atanh_dt = atanh_taylor13(dt);
        s+=(atanh_cc-atanh_dt)*(atanh_cc-atanh_dt)*w;
        for (size_t k=0;k!=D;++k){
          //grad(i*D+k)+=2*x(j*D+k)*(atanh_dt-atanh_cc)/(v*(1-dt*dt));
          //grad(j*D+k)+=2*x(i*D+k)*(atanh_dt-atanh_cc)/(v*(1-dt*dt));
          grad(i*D+k)+=2*x(j*D+k)*diff_atanh_taylor13(dt)*(atanh_dt-atanh_cc)*w;
          grad(j*D+k)+=2*x(i*D+k)*diff_atanh_taylor13(dt)*(atanh_dt-atanh_cc)*w;
        }
      }
    }
    cerr << s << endl;
    return s;
  }
};


int main(int argc, char *argv[])
{
  vector<unordered_map<size_t,array<double,2>>> cctable;
  while (cin){
    size_t i,j,n;
    double cc;
    cin >> i >> j >> cc >> n ;
    if (i<j) swap(i,j);
    if (i>=cctable.size()) cctable.resize(i+1);
    cctable[i][j]={cc,n-3.0};
  }
  cerr << cctable.size() << " points" << endl;
  LBFGSpp::LBFGSParam<double> param;
  param.epsilon = 1e-4;
  param.max_iterations = 256;
  LBFGSpp::LBFGSSolver<double> solver(param);
  Eigen::VectorXd x(D*cctable.size());
  std::mt19937_64 mt;
  std::uniform_real_distribution<double> nd(-1,1);
  for (size_t i=0;i!=cctable.size();++i){
    double d=2;
    while (d>=0.9){
      for (size_t j=0;j!=D;++j) x(i*D+j)=nd(mt);
      d=0;
      for (size_t j=0;j!=D;++j) d+=x(i*D+j)*x(i*D+j);
    }
  }
  double fx=0;
  alg2_1_target target{cctable};
  /*Eigen::VectorXd grad(D*cctable.size());
  const double v0 = target(x,grad);
  for (size_t i=0;i!=cctable.size();++i){ 
    Eigen::VectorXd y = x;
    Eigen::VectorXd grad2(D*cctable.size());
    y[i]+=1e-4;
    const double v1 = target(y,grad2);
    if((grad(i)-(v1-v0)*1e4)*(grad(i)-(v1-v0)*1e4)/((grad(i)+(v1-v0)*1e4)*(grad(i)+(v1-v0)*1e4))>0.01)
    cerr << i << " " << grad(i) << " " << (v1-v0)*1e4 << " " << v1 << " " << v0 << endl;
  }
  return 1;*/
  solver.minimize(target,x,fx);
  for (size_t i=0;i!=cctable.size();++i){
    for (size_t j=0;j!=D;++j){
      cout << x(i*D+j) << " ";
    }
    cout << endl;
  }
  return 0;
}
