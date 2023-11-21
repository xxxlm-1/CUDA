# [加速计算基础——CUDA C](https://learn.next.courses.nvidia.com/courses/course-v1:DLI+C-AC-01+V1-ZH/course/#block-v1:DLI+C-AC-01+V1-ZH+type@chapter+block@85f2a3ac16a0476685257996b84001ad)

---

[TOC]

----

## 第一节

#### **1.1 目标**

> - 编写、编译及运行既可调用 CPU 函数也可**启动** GPU **核函数** 的 C/C++ 程序。
> - 使用**执行配置**控制并行**线程层次结构**。
> - 重构串行循环以在 GPU 上并行执行其迭代。
> - 分配和释放可用于 CPU 和 GPU 的内存。
> - 处理 CUDA 代码生成的错误。
> - 加速 CPU 应用程序。



#### 1.1.1加速系统

又称\*异构系统\*，由 CPU 和 GPU 组成。

![image-20231111153836979](D:\My-Study-App\CUDA\CUDA-pictures\image-20231111153836979.png)



#### 1.1.2 改CPU函数为 GPU函数

1.函数名   __global__

```
__global__ void GPUFunction()
```

2.指定调用GPU的块与线程数

```
GPUFunction<<<线程数, 线程块数>>>();
```

3.将GPU与CPU运行的同步

```
cudaDeviceSynchronize();
```

> 1.在 CPU 上执行的代码称为**主机**代码，而将在 GPU 上运行的代码称为**设备**代码。
>
> 2.CPU函数执行比GPU函数快.

实操：[01.01-hello-gpu.cu](https://github.com/xxxlm-1/CUDA/blob/main/chapter_one/01.01-hello-gpu.cu)



#### 1.1.3 CUDA的线性层次结构

1.`gridDim.x`		网格中的线程块数

2.`threadIdx.x`	 网格中线程（位于线程块内）的索引 

3.`blockDim.x`		网格中线程块的线程数

 4.`blockIdx.x`		网格中线程块（位于网格内）的索引

> 1. 与给定核函数启动相关联的块的集合称为网格；
> 2. 索引从0开始；

实操 ：

​		[03.01-thread-and-block-idx (1).cu](https://github.com/xxxlm-1/CUDA/blob/main/chapter_one/03.01-thread-and-block-idx%20(1).cu)

​		[04.01-single-block-loop.cu](https://github.com/xxxlm-1/CUDA/blob/main/chapter_one/04.01-single-block-loop.cu)



#### 1.1.4 协调并行线程

 `threadIdx.x + blockIdx.x * blockDim.x`

> 线程块包含的线程具有数量限制：确切地说是 1024 个。

实操：[05.02-multi-block-loop.cu](https://github.com/xxxlm-1/CUDA/blob/main/chapter_one/05.02-multi-block-loop.cu)  练习：加速具有多个线程块的For循环



#### 1.1.5 动态在GPU上分配内存

`cudaMallocManaged()`  替换  ` malloc()`

`cudaFree()`  替换   `free()`

```c
int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
cudaMallocManaged(&a, size);
cudaFree(a);
```

实操：[06.01-double-elements.cu](https://github.com/xxxlm-1/CUDA/blob/main/chapter_one/06.01-double-elements.cu)   练习：主机和设备上的数组操作



#### 1.1.6 网格大小与工作量不匹配

##### 1.1.6 .1 线程数超过任务

` threadIdx.x + blockIdx.x * blockDim.x` = `dataIndex` < N

```
__global__ some_kernel(int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < N) 
  // Check to make sure `idx` maps to some value within `N`
  {
    // Only do work if it does
  }
}
```

实操：[07.02-mismatched-config-loop.cu](https://github.com/xxxlm-1/CUDA/blob/main/chapter_one/07.02-mismatched-config-loop.cu) 练习：使用不匹配的执行配置来加速For循环



##### 1.1.6.2  任务超过线程数  （跨网格的循环）

```c
__global void kernel(int *a, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    // do work on a[i];
  }
}
```



实操：[08.03-grid-stride-double.cu](https://github.com/xxxlm-1/CUDA/blob/main/chapter_one/08.03-grid-stride-double.cu)   练习：使用跨网格循环来处理比网格更大的数组



#### 1.3 错误处理

> 有许多 CUDA 函数（例如，内存管理函数）会返回类型为 `cudaError_t` 的值，该值可用于检查调用函数时是否发生错误,以下是对调用 `cudaMallocManaged` 函数执行错误处理的示例：

```c
cudaError_t err;
err = cudaMallocManaged(&a, N)                    // Assume the existence of `a` and `N`.
if (err != cudaSuccess)                           // `cudaSuccess` is provided by CUDA.
{
  printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString` is provided by CUDA.
}
```

> 启动定义为返回 `void` 的核函数,为检查启动核函数时是否发生错误（例如，如果启动配置错误），CUDA 提供 `cudaGetLastError` 函数，该函数会返回类型为 `cudaError_t` 的值。

```c
/*
 * This launch should cause an error, but the kernel itself
 * cannot return it.
 */
someKernel<<<1, -1>>>();  // -1 is not a valid number of threads.

cudaError_t err;
err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
if (err != cudaSuccess)
{
  printf("Error: %s\n", cudaGetErrorString(err));
}
```

创建一个包装 CUDA 函数调用的宏

```c
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int main()
{
/*
 * The macro can be wrapped around any function returning
 * a value of type `cudaError_t`.
 */
  checkCuda( cudaDeviceSynchronize() )
}
```

实操: [09.01-add-error-handling.cu](https://github.com/xxxlm-1/CUDA/blob/main/chapter_one/09.01-add-error-handling.cu)	练习：添加错误处理



#### 1.4 最后练习 

[10.01-vector-add.cu](https://github.com/xxxlm-1/CUDA/blob/main/chapter_one/10.01-vector-add.cu)   练习:加速向量加法 



#### 1.5 进阶 2维和3维

```c
dim3 threads_per_block(16, 16, 1);
dim3 number_of_blocks(16, 16, 1);
someKernel<<<number_of_blocks, threads_per_block>>>();
鉴于以上示例，someKernel 内部的变量 
    gridDim.x、gridDim.y、blockDim.x 和 blockDim.y 均将等于 16。
    
```

   [11.01-matrix-multiply-2d.cu](https://github.com/xxxlm-1/CUDA/blob/main/chapter_one/11.01-matrix-multiply-2d.cu) 练习：加速2D矩阵乘法应用

​	[12.01-heat-conduction.cu](https://github.com/xxxlm-1/CUDA/blob/main/chapter_one/12.01-heat-conduction.cu)    练习：给热传导应用程序加速

```c
 dim3 blocks(32,16,1);
 dim3 grids((nj/blocks.x)+1,(ni/blocks.y)+1,1);
```

