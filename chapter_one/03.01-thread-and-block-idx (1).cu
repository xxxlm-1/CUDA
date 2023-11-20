#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  if(threadIdx.x == 15 && blockIdx.x == 7)
  {
    printf("Success!\n");
  } else {
    printf("Failure. Update the execution configuration as necessary.\n");
  }
}

int main()
{
  printSuccessForCorrectExecutionConfiguration<<<8,16>>>();
  cudaDeviceSynchronize();
}