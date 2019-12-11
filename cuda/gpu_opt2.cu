#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f
#define G 6.674083131313131313e-11
#define SOLAR_MASS 1.989e13

typedef struct { float4 *pos, *newvel, *oldvel; } Body;

void gen_body_data(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void update_kernel(float4 *p, float4 *v,float4 *u,float *m, float dt, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float Ax = 0.0f; float Ay = 0.0f; float Az = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) {
      __shared__ float3 spos[BLOCK_SIZE];
      float4 tpos = p[tile * blockDim.x + threadIdx.x];
      spos[threadIdx.x] = make_float3(tpos.x, tpos.y, tpos.z);
      __syncthreads();

      #pragma unroll
      for (int j = 0; j < BLOCK_SIZE; j++) {
        //  3 flops
        float dx = spos[j].x - p[i].x;
        float dy = spos[j].y - p[i].y;
        float dz = spos[j].z - p[i].z;

        float dis_sqr = dx*dx + dy*dy + dz*dz + SOFTENING;  // 6 flops
        
        float magnitude = rsqrtf(dis_sqr);                  // 2 flops
        float mag_cube = magnitude * magnitude * magnitude; // 2 flops

        Ax += m[j] * dx * mag_cube;                         // 6 flops
        Ay += m[j] * dy * mag_cube;
        Az += m[j] * dz * mag_cube;
      }
      __syncthreads();
    }

    v[i].x += dt*Ax + v[i].x;
    v[i].y += dt*Ay + v[i].y;
    v[i].z += dt*Az + v[i].z;
  }
}

int main(const int argc, const char** argv) {

  int num_body = 10000000; // 10 million
  if (argc > 1) num_body = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int num_time_steps = 10;  // simulation iterations

  int bytes = 3*num_body*sizeof(float4);
  int mass_size = num_body * sizeof(float);
  float *buf = (float*)malloc(bytes);
  float *mass = NULL;
  Body p = { (float4*)buf, ((float4*)buf) + num_body , ((float4*)buf) + num_body + num_body};

  gen_body_data(buf, 12*num_body); // Init pos / vel data

  cudaMallocHost((void **)&mass, num_body * sizeof(float));

  for(int l = 0;l<num_body;l++){
    mass[l] = SOLAR_MASS * G * (rand() / (float)RAND_MAX); // generating the mass values
  }
  float *d_mass;
  cudaMalloc((void **)&d_mass, num_body * sizeof(float));

  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  Body d_p = { (float4*)d_buf, ((float4*)d_buf) + num_body, ((float4*)d_buf) + num_body + num_body };

  int nBlocks = (num_body + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double totalTime = 0.0;

  for (int iter = 1; iter <= num_time_steps; iter++) {


    cudaMemcpyAsync(d_mass, mass, mass_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
                                                             // Timing the calculations
    StartTimer();
    update_kernel<<<nBlocks, BLOCK_SIZE>>>(d_p.pos, d_p.newvel, d_p.oldvel, d_mass, dt, num_body);
    cudaDeviceSynchronize();
    const double tElapsed = GetTimer() / 1000.0;
    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0 ; i < num_body; i++) { // Assigning the newly calculated position
        p.pos[i].x += (p.newvel[i].x)*dt;
        p.pos[i].y += (p.newvel[i].y)*dt;
        p.pos[i].z += (p.newvel[i].z)*dt;
      }
      printf("In %d timestep, position cordinates: %f\t %f\t %f\n",iter, p.pos[0].x , p.pos[0].y , p.pos[0].z);
    if (iter > 1) { // Neglecting the first iteration
      printf("In %d iteration: %lf\n",iter, tElapsed);
      totalTime += tElapsed;
    }

}
double avgTime = totalTime / (double)(num_time_steps-1);

double float_ops_per_interaction =19 + (9/ num_body) ;

double expected_time = float_ops_per_interaction/2.91e12; // gpu peak flop rate is 2.91e12 for Tesla K80

printf("expected time: %0.9lf\t Average time:%0.9lf\n", expected_time*num_body*num_body, avgTime);
printf("Bodies: %d Expected: %0.3lf billion interactions / second\n", num_body,(1e-9 ) /expected_time);
printf("Bodies: %d average %0.3lf billion interactions / second\n\n", num_body, (1e-9*num_body*num_body ) / avgTime);

free(buf);
cudaFree(d_buf);
cudaFree(d_mass);
cudaFreeHost(mass);
}
