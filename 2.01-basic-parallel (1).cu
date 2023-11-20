#include <stdio.h>

__global__ void firstParallel()
{
  printf("This should be running in parallel.\n");
}

int main()
{
  firstParallel<<<1,5>>>();
  cudaDeviceSynchronize();
}
/*
This should be running in parallel.
This should be running in parallel.
This should be running in parallel.
This should be running in parallel.
This should be running in parallel.
*/