#pragma once
#include <math.h>
// Loss = T^2 KL(Q || softmax(z/T)); derivative dL/dz = T (p_T - Q).
// Caller guarantees Q is a finite, normalized distribution.
namespace KDMath {
inline bool lossGradient(const float *z, const float *q, int n, float temperature,
                         float *gradient, float &loss) {
    if (!z || !q || !gradient || n <= 0 || !isfinite(temperature) || temperature <= 0.f) return false;
    float m = -INFINITY, qsum = 0.f;
    for (int k=0;k<n;k++) {
        if (!isfinite(z[k]) || !isfinite(q[k]) || q[k] < 0.f) return false;
        float a=z[k]/temperature;
        if (a>m) m=a;
        qsum+=q[k];
    }
    if (!isfinite(qsum) || fabsf(qsum-1.f)>0.005f) return false;
    float denom=0.f;
    for(int k=0;k<n;k++) { gradient[k]=expf(z[k]/temperature-m); denom+=gradient[k]; }
    if (!isfinite(denom) || denom<=0.f) return false;
    loss=0.f;
    for(int k=0;k<n;k++) {
        float p=gradient[k]/denom;
        if(q[k]>0.f) loss+=q[k]*(logf(q[k])-logf(fmaxf(p,1.0e-30f)));
        gradient[k]=temperature*(p-q[k]);
    }
    loss*=temperature*temperature;
    return isfinite(loss);
}
}
