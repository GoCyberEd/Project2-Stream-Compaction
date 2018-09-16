#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 256

namespace StreamCompaction {
    namespace Efficient {

    	int * dev_idata;
    	int * dev_odata;

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int pow, int pow1, int data[]) {
        	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
        	if (idx > n) {
        		return;
        	}
        	if (idx * pow1 > n){
        		return;
        	}

        	data[(idx + 1) * pow1 - 1] += data[(idx + 1) * pow1 - 1 - pow];
        }

        __global__ void kernDownSweep(int n, int pow, int pow1, int data[]) {
        	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
        	if (idx > n) { return; }

        	int aux = data[pow1];
        	data[pow1] += data[pow1 - pow];
        	data[pow1 - pow] = aux;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
        	int logValue = ilog2ceil(n);
        	int pown = std::pow(2, logValue);

        	cudaMalloc((void**) &dev_idata, pown * sizeof(int));
        	cudaMemset(dev_idata, 0, n * sizeof(int));
        	cudaMalloc((void**) &dev_odata, pown * sizeof(int));

        	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
        	//cudaMemcpy(dev_odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

        	int pow, pow1, blocksPerGrid;

            timer().startGpuTimer();
            // Need two separate kernels, one for upsweep and one for down to ensure everything stays in sync
            // Can we just use sync_threads? No, becaue potentially multiple blocks
            // 1. upsweep (note it updates in place, hopefully this is okay? Just summing)
            // 2. Reset end of array to 0
            // 3. Downsweep
            for (int d = 0; d < logValue; d++) {
            	pow = std::pow(2, d);
            	pow1 = std::pow(2, d + 1);
            	blocksPerGrid = (pown / pow1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            	kernUpSweep<<< blocksPerGrid, BLOCK_SIZE >>>(pown, pow, pow1, dev_idata);
            }
            for (int d = 0; d < logValue; d++) {
            	pow = std::pow(2, d);
            	pow1 = std::pow(2, d + 1);
            	blocksPerGrid = (pown / pow1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            	kernDownSweep<<< blocksPerGrid, BLOCK_SIZE >>>(pown, pow, pow1, dev_idata);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
