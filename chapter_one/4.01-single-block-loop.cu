#include <stdio.h>

__global__ void loop(int N)
{
  for (int i = 0; i < N; ++i)
  {
    printf("This is iteration number %d\n", i);
  }
}

int main()
{
  int N = 10;
  loop<<<1,1>>>(N);
  cudaDeviceSynchronize();
}