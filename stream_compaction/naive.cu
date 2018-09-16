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
        	if (idx > n) {
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
        	cudaMalloc((void**) &naive_odata, n * sizeof(int));

        	//Transfer from host memory to device
        	cudaMemcpy(naive_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

        	int blocksPerGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            timer().startGpuTimer();
        	int logValue = ilog2ceil(n);
        	int pow;

        	for (int d = 0; d < logValue; d++){
        		pow = std::pow(2, d);
        		kernNaiveScan<<< blocksPerGrid, BLOCK_SIZE >>>(n, pow, naive_odata, naive_idata);

        		std::swap(naive_odata, naive_idata);
        	}
            timer().endGpuTimer();

            // Copy back to host, idata because of swap
            cudaMemcpy(odata, naive_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(naive_idata);
            cudaFree(naive_odata);
        }
    }
}
