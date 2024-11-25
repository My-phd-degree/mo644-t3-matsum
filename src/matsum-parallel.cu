#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

__global__ void matrix_sum(int *_C, int *_A, int *_B, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size)
    _C[i] = _A[i] + _B[i];
}

#define CUDACHECK(cmd) \
  do { \
    cudaError_t e = cmd; \
    if ( e != cudaSuccess) { \
      printf("Failed: Cuda error %s:%s '%s'\n", \
          __FILE__, __LINE__, \
          cudaGetErrorString(e)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0);


int main(int argc, char **argv) {
  //host variables
  int *A, 
      *B, 
      *C;
  int i, 
      j;
  double t;
  int size;
  int bytes;

  //device variables
  int *_A, 
      *_B, 
      *_C;

  // Input
  int rows, 
      cols;
  FILE *input;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return EXIT_FAILURE;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open file\n");
    return EXIT_FAILURE;
  }

  fscanf(input, "%d", &rows);
  fscanf(input, "%d", &cols);

  size = rows * cols;
  bytes = sizeof(int) * size;

  // Allocate memory on the host
  A = (int *)malloc(bytes);
  B = (int *)malloc(bytes);
  C = (int *)malloc(bytes);

  // Allocate memory on the device
  CUDACHECK(cudaMalloc(&_A, bytes));
  CUDACHECK(cudaMalloc(&_B, bytes));
  CUDACHECK(cudaMalloc(&_C, bytes));
  
  // Initialize memory
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      A[i * cols + j] = B[i * cols + j] = i + j;
    }
  }

  // Copy data to device
  CUDACHECK(cudaMemcpy(_A, A, bytes, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(_B, B, bytes, cudaMemcpyHostToDevice));

  // Compute matrix sum on device
  // Leave only the kernel and synchronize inside the timing region!
  int nBlocks = (size + 127)/128;
  t = omp_get_wtime();
  matrix_sum<<<nBlocks, 128>>>(_C, _A, _B, size);
  CUDACHECK(cudaDeviceSynchronize());
  t = omp_get_wtime() - t;

  // Copy data back to host
  CUDACHECK(cudaMemcpy(C, _C, bytes, cudaMemcpyDeviceToHost));

  long long int sum = 0;

  // Keep this computation on the CPU
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      sum += C[i * cols + j];
    }
  }

  fprintf(stdout, "%lli\n", sum);
  fprintf(stderr, "%lf\n", t);

  free(A);
  free(B);
  free(C);
  CUDACHECK(cudaFree(_A));
  CUDACHECK(cudaFree(_B));
  CUDACHECK(cudaFree(_C));
}
