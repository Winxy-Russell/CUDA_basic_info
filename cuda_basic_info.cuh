#include <cuda_runtime.h>
#include <stdio.h>


#define CHECK(call){ \
    const cudaError_t error = call; \
    if(error != cudaSuccess){      \
        printf("Error: %s:%d, ",__FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10 * error);      \
        }                     \
}

int dev = 0;

int getDeviceCount(){
    int deviceCnt = 0;
    CHECK(cudaGetDeviceCount(&deviceCnt));
    if(!deviceCnt){
        printf("There are no available device that support CUDA\n");
    }
    else{
        printf("Detect %d CUDA devices\n", deviceCnt);
    }
    return deviceCnt;
}

int setReturnBestDevice(){
    int device_cnt = getDeviceCount();
    int maxMultiprocessors = 0, dev = 0;
    if(device_cnt > 1){
        for(int device = 0;device < device_cnt;device++){
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device);
            if(maxMultiprocessors < props.multiProcessorCount){
                maxMultiprocessors = props.multiProcessorCount;
                dev = device;
            }
        }
        cudaSetDevice(dev);
    }
    return dev;
}

void getMemory(){ // assumed that the device has been selected
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev);
    printf("Total amount of global memory:\t %.2f GBytes (%llu bytes)\n", (float)props.totalGlobalMem / pow(1024.0, 3), (unsigned long long)props.totalGlobalMem);
    if(props.l2CacheSize){
        printf("l2 cache size:\t %d bytes\n", props.l2CacheSize);
    }
    printf("Total amount of constant memory:\t %lu bytes\n", (unsigned long)props.totalConstMem);
    printf("Total amount of shared memory per block:\t %lu\n", (unsigned long)props.sharedMemPerBlock);
    printf("Total number of registers per block:\t %d\n", props.regsPerBlock);
    printf("Maximum memory pitch:\t %lu bytes\n", (unsigned long)props.memPitch);
}

void getInfoOfthreadsAndBlocks(){
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev);

}

void flow(){
    getDeviceCount();
    setReturnBestDevice();
    getMemory();
}