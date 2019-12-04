#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f
#define G 6.674083131313131313e-11
#define SOLAR_MASS 1.989e30

typedef struct { float4 *pos, *newvel, *oldvel; } BodySystem;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void bodyForce(float4 *p, float4 *v,float4 *u,float *m, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Ax = 0.0f; float Ay = 0.0f; float Az = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;

      float mg = G * m[j];

      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Ax += mg * dx * invDist3;
      Ay += mg * dy * invDist3; 
      Az += mg * dz * invDist3;
    }

    v[i].x += dt*Ax + u[i].x;
    v[i].y += dt*Ay + u[i].y;
    v[i].z += dt*Az + u[i].z;
  }
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);
  
  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations
  
  int bytes = 3*nBodies*sizeof(float4);
  int mass_size = nBodies * sizeof(float);
  float *buf = (float*)malloc(bytes);
  float *mass = new float[nBodies];
  BodySystem p = { (float4*)buf, ((float4*)buf) + nBodies };

  randomizeBodies(buf, 12*nBodies); // Init pos / vel data

  cudaMallocHost((void **)&mass,mass_size);
  for(int l = 0;l<nBodies;l++){
    mass[l] = SOLAR_MASS * (rand() / (float)RAND_MAX);
  }

  float *d_mass;
  cudaMalloc((void **)&d_mass, mass_size));

  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  BodySystem d_p = { (float4*)d_buf, ((float4*)d_buf) + nBodies };

  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double totalTime = 0.0; 

  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();

    cudaMemcpy(d_mass, mass, mass_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p.pos, d_p.newvel, d_p.oldvel, d_mass, dt, nBodies);
    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p.pos[i].x += 0.5*(p.newvel[i].x + p.oldvel[i].x)*dt;
      p.pos[i].y += 0.5*(p.newvel[i].y + p.oldvel[i].y)*dt;
      p.pos[i].z += 0.5*(p.newvel[i].z + p.oldvel[i].z)*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }
#ifndef SHMOO
    printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
#endif
  }
  double avgTime = totalTime / (double)(nIters-1); 

  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);


  free(buf);
  cudaFree(d_buf);
  cudaFree(d_mass);
  cudaFreeHost(mass);
}