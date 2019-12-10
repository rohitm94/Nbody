#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define BLOCK_SIZE 512
#define SOFTENING 1e-9f
#define G 6.674083131313131313e-11
#define SOLAR_MASS 1.989e30

typedef struct { float x, y, z, vx, vy, vz, ux, uy, uz, m; } Body;

void gen_body_data(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}
 
__global__
void update_kernel(Body *p, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Ax = 0.0f; float Ay = 0.0f; float Az = 0.0f;

    for (int j = 0; j < n; j++) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;

        float mg = G * p[j].m ;

        float dis_sqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float magnitude = rsqrtf(dis_sqr);
        float mag_cube = magnitude * magnitude * magnitude;

        Ax += mg * dx * mag_cube;
        Ay += mg * dy * mag_cube;
        Az += mg * dz * mag_cube;
    }
    p[i].vx += dt*Ax + p[i].ux;
    p[i].vy += dt*Ay + p[i].uy;
    p[i].vz += dt*Az + p[i].uz;
  }
}

int main(const int argc, const char** argv) {
  
  int num_body = 10000000; //10 million
  if (argc > 1) num_body = atoi(argv[1]);
  
  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = num_body*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  gen_body_data(buf, 10*num_body); // Init pos / vel data

  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  Body *d_p = (Body*)d_buf;

  int nBlocks = (num_body + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double totalTime = 0.0; 

  for (int iter = 1; iter <= nIters; iter++) {

    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
    StartTimer();
    update_kernel<<<nBlocks, BLOCK_SIZE>>>(d_p, dt, num_body); 
    cudaDeviceSynchronize();
    const double tElapsed = GetTimer() / 1000.0;
    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0 ; i < num_body; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }
    
    if (iter > 1) { // First iter is warm up
      printf("Iteration %d: %.3lf seconds\n", iter, tElapsed);
      totalTime += tElapsed; 
    }

  }
  double avgTime = totalTime / (double)(nIters-1); 

  double float_ops_per_interaction =22 + (9/ num_body) ;

  double expected_time = float_ops_per_interaction/2.19e12;

  printf("expected time: %0.9lf\t Average time:%0.9lf\n", expected_time*num_body*num_body, avgTime);
  printf("Bodies: %d Expected: %0.3lf billion interactions / second\n", num_body,(1e-9 ) /expected_time);
  printf("Bodies: %d average %0.3lf billion interactions / second\n\n", num_body, (1e-9*num_body*num_body ) / avgTime);
  
  free(buf);
  cudaFree(d_buf);
}
