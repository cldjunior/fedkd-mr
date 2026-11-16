#include "include/KDMath.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>
static float loss(const float* z,const float* q,int n,float t){float grad[8], l=0;assert(KDMath::lossGradient(z,q,n,t,grad,l));return l;}
int main(){
 const float temps[]={0.7f,1.f,2.f,3.f,4.f};
 for(float t:temps){
  float z[]={-2.2f,0.3f,1.4f,4.f},q[]={0.1f,0.2f,0.6f,0.1f},g[4],l=0;
  assert(KDMath::lossGradient(z,q,4,t,g,l));assert(l>=0.f);
  for(int i=0;i<4;++i){float old=z[i],h=0.002f;z[i]=old+h;float hi=loss(z,q,4,t);z[i]=old-h;float lo=loss(z,q,4,t);z[i]=old;float fd=(hi-lo)/(2*h);if(fabsf(fd-g[i])>0.003f){printf("FAIL T=%.2f class=%d got=%f expected=%f\n",t,i,g[i],fd);return 1;}}
 }
 float z[]={0,0,0,0},q[]={0.25,0.25,0.25,0.25},g[4],l;
 assert(KDMath::lossGradient(z,q,4,2.f,g,l));assert(fabsf(l)<1e-6f);
 float bad[]={0.5f,0.5f,0.f,0.5f};assert(!KDMath::lossGradient(z,bad,4,2.f,g,l));
 puts("PASS: KD C++ T^2 KL, gradient finite differences, T=0.7/1/2/3/4, invalid teacher");
}
