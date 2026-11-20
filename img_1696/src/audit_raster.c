/* Orthographic inspection rasterizer, depth-tested and clipped per sample. */
#include <math.h>
#include <stdint.h>
void draw_mesh(int w,int h,int nf,const float *v,const int *f,const uint8_t *color,
               const int *owner,const uint8_t *keep,const float *world,int cutaway,float near_d,float far_d,
               float *depth,int *ids,uint8_t *rgb){
    for(int i=0;i<nf;i++){
        if(!keep[i])continue;
        const float *p0=v+3*f[3*i],*p1=v+3*f[3*i+1],*p2=v+3*f[3*i+2];
        float xmin=fminf(p0[0],fminf(p1[0],p2[0])),xmax=fmaxf(p0[0],fmaxf(p1[0],p2[0]));
        float ymin=fminf(p0[1],fminf(p1[1],p2[1])),ymax=fmaxf(p0[1],fmaxf(p1[1],p2[1]));
        int a=(int)fmaxf(0,floorf(xmin)),b=(int)fminf(w-1,ceilf(xmax));
        int c=(int)fmaxf(0,floorf(ymin)),d=(int)fminf(h-1,ceilf(ymax));
        float s0=p1[0]-p0[0],s1=p1[1]-p0[1],t0=p2[0]-p0[0],t1=p2[1]-p0[1];
        float det=s0*t1-s1*t0;if(fabsf(det)<1e-5)continue;
        for(int y=c;y<=d;y++){
            for(int x=a;x<=b;x++){
                float qx=x+.5f-p0[0],qy=y+.5f-p0[1];
                float u=(qx*t1-qy*t0)/det,vv=(s0*qy-s1*qx)/det;
                if(u<-.00005f||vv<-.00005f||u+vv>1.00005f)continue;
                float z=p0[2]+u*(p1[2]-p0[2])+vv*(p2[2]-p0[2]);
                if(z<near_d||z>far_d)continue;
                if(cutaway){
                    const float *w0=world+3*f[3*i],*w1=world+3*f[3*i+1],*w2=world+3*f[3*i+2];
                    float wy=w0[1]+u*(w1[1]-w0[1])+vv*(w2[1]-w0[1]);
                    float wx=w0[0]+u*(w1[0]-w0[0])+vv*(w2[0]-w0[0]);
                    if(wy<.25f||wy>8.42f||wx<-.15f||wx>4.98f)continue;
                }
                int j=y*w+x;
                if(z<=depth[j])continue;
                depth[j]=z;ids[j]=owner[i];
                for(int k=0;k<3;k++)rgb[3*j+k]=color[3*i+k];
            }
        }
    }
}
