#include <stdio.h>
__global__ void loop(int N)
{
    int i= threadIdx.x   + blockDim.x * blockIdx.x ;
    printf("%d  ", i);
}

int main()
{
  int N = 10;
  loop<<<2,5>>>(N);
  cudaDeviceSynchronize();
}