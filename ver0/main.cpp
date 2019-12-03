
#include <iostream>

#include "GSimulation.hpp"

int main(int argc, char **argv)
{
  int N;     //number of particles
  int nstep; //number ot integration steps

  GSimulation sim;

  if (argc > 1)
  {
    N = atoi(argv[1]);
    sim.set_number_of_particles(N);
    if (argc == 3)
    {
      nstep = atoi(argv[2]);
      sim.set_number_of_steps(nstep);
    }
  }

  sim.start();

  return 0;
}
