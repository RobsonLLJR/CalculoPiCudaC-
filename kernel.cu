#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif
#define BLOQUE 1
#define HILOS 512

__device__ float calcularArea(float inicio, float final, float base)
{
    float medio = (inicio + final) / 2;
    float altura = 4 / (1 + (medio) * (medio));
    return base * altura;
}

__global__ void calcularPi(float* pi, int* precisaoEscolhida)
{
    int lancamentos = *precisaoEscolhida;
    int identificador = threadIdx.x;
    float inicio;
    float final;
    extern __shared__ float area[];
    float superVar = 1 / (float)lancamentos;
    cuda_SYNCTHREADS();

    int* array;
    if (lancamentos % 2 != 0)
    {
        if (identificador == (lancamentos - 1))
        {
            array[0] += array[identificador];
        }
    }
    cuda_SYNCTHREADS();

    int salto = lancamentos / 2;

    while (salto)
    {
        if (identificador < salto)
        {
            area[identificador] = array[0] + array[identificador];
        }
    }
    cuda_SYNCTHREADS();

    if (identificador == 0)
    {
        *pi = area[identificador];
    }
}

__host__ int clean_stdin(void)
{
    while (getchar() != '\n')
        ;
    return 1;
}

int main(int argc, char** argv)
{
    float* dev_pi, * hst_pi;
    int* dev_precition;
    cudaSetDevice(0);        
    cudaEvent_t start, stop; 
    int precition;
    char c;
    char linea[] = "---------------------------------------------------------------------";
    cudaDeviceProp features;               
    cudaGetDeviceProperties(&features, 0);
    
    do
    {
        printf("Qual precisao deseja para calcular PI, maximo %d: ", features.maxThreadsPerBlock);
        //
        if (scanf("%d%c", &precition, &c) != 2 || c != '\n')
        {
            printf("Valor invalido\n");
            clean_stdin();
        }
    } while (precition < 0 || precition > features.maxThreadsPerBlock);
    //Reservando memoria
    hst_pi = (float*)malloc(sizeof(float));
    cudaMalloc((void**)&dev_pi, sizeof(float));
    cudaMalloc((void**)&dev_precition, sizeof(int));
    cudaMemcpy(dev_precition, &precition, sizeof(int), cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    calcularPi <<< BLOQUE, precition, precition * sizeof(float) >>> (dev_pi, dev_precition);
    cudaMemcpy(hst_pi, dev_pi, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    //Impressao dos resultados
    printf("\n%s\nmemoria compartilhada disponivel %d KiB:\n", linea, features.sharedMemPerBlock / 1024);
    printf("Valo de PI calculado: %f\n", *hst_pi);
    printf("Tempo de execucao \n", elapsedTime);
    getchar();
    return 0;
}
