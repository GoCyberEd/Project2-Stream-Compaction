#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 256

namespace StreamCompaction {
    namespace Naive {
    	int* naive_idata;
    	int* naive_odata;

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // __global__
        __global__ void kernNaiveScan(int n, int pow, int * odata, int * idata) {
        	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
        	if (idx >= n) {
        		return;
        	}

        	if (idx >= pow) {
        		odata[idx] = idata[idx - pow] + idata[idx];
        	} else {
        		odata[idx] = idata[idx];
        	}

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
        	cudaMalloc((void**) &naive_idata, n * sizeof(int));
        	checkCUDAError("Failed to allocate idata");
        	cudaMalloc((void**) &naive_odata, n * sizeof(int));
        	checkCUDAError("Failed to allocate odata");

        	//Transfer from host memory to device
        	cudaMemcpy(naive_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
        	checkCUDAError("cudaMemcpy failed (initial)");

        	int blocksPerGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            timer().startGpuTimer();
        	int logValue = ilog2ceil(n);
        	int pow;
        	for (int d = 1; d <= logValue; d++){
        		pow = 1 << d - 1;
        		kernNaiveScan<<< blocksPerGrid, BLOCK_SIZE >>>(n, pow, naive_odata, naive_idata);
            	checkCUDAError("kernNaiveScan failed");

        		std::swap(naive_odata, naive_idata);
        	}
        	//std::swap(naive_odata, naive_idata);

        	Common::kernConvertScanToExclusive<<< blocksPerGrid, BLOCK_SIZE >>>(n, naive_odata, naive_idata);
        	checkCUDAError("kernConvertScanToExclusive failed");
            timer().endGpuTimer();

            cudaMemcpy(odata, naive_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy failed (final)");

            cudaFree(naive_idata);
        	checkCUDAError("cudaFree idata failed");
            cudaFree(naive_odata);
            checkCUDAError("cudaFree odata failed");
        }
    }
}
