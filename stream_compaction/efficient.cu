#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {

    	int * dev_data;

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int pow, int pow1, int data[]) {
        	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
        	if (idx >= n) {
        		return;
        	}

        	int i = idx * pow1;
        	if (i < n) {
        		data[i + pow1 - 1] += data[i + pow - 1];
        	}
        }

        __global__ void kernDownSweep(int n, int pow, int pow1, int data[]) {
        	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
        	if (idx >= n) { return; }

        	int i = idx * pow1;
        	if (i < n) {
        		// Swap and sum
        		int aux = data[i + pow - 1];
        		data[i + pow - 1] = data[i + pow1 - 1];
        		data[i + pow1 - 1] += aux;
        	}
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
        	int ceil = ilog2ceil(n);
        	int ceilN = 1 << ceil;

        	cudaMalloc((void**) &dev_data, ceilN * sizeof(int));
        	checkCUDAError("malloc dev_data failed");
        	cudaMemset(dev_data, 0, n * sizeof(int));
        	checkCUDAError("cudaMemset to clear array failed");

        	cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
        	checkCUDAError("cudaMemcpy input host to device failed");
        	//cudaMemcpy(dev_odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

        	int pow, pow1, blocksPerGrid;

            timer().startGpuTimer();
            // Need two separate kernels, one for upsweep and one for down to ensure everything stays in sync
            // Can we just use sync_threads? No, becaue potentially multiple blocks
            // 1. upsweep (note it updates in place, hopefully this is okay? Just summing)
            // 2. Reset end of array to 0
            // 3. Downsweep
            for (int d = 0; d < ceil; d++) {
            	//pow = std::pow(2, d);
            	//pow1 = std::pow(2, d + 1);
            	pow = 1 << d;
            	pow1 = 1 << (d + 1);
            	blocksPerGrid = (ceilN / pow1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            	kernUpSweep<<< blocksPerGrid, BLOCK_SIZE >>>(ceilN, pow, pow1, dev_data);
            	checkCUDAError("kernUpSweep failed");
            }

            // Reset last value
            //int z = 0;
            //cudaMemcpy(dev_data + ceilN - 1, &z, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_data + ceilN - 1, 0, sizeof(int));
            checkCUDAError("cudaMemcpy zero failed");
            //dev_data[ceilN - 1] = 0;

            //for (int d = 0; d < ceil; d++) { start at end instead
            for (int d = ceil - 1; d >= 0; d--){
            	pow = 1 << d;
            	pow1 = 1 << (d + 1);
            	blocksPerGrid = (ceilN / pow1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            	kernDownSweep<<< blocksPerGrid, BLOCK_SIZE >>>(ceilN, pow, pow1, dev_data);
            	checkCUDAError("kernDownSweep failed");
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("memcpy answer to host failed");

            cudaFree(dev_data);
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
            int * dev_bools;
            int * dev_indices;
            int * dev_scatter;
            int * dev_input;
            cudaMalloc((void**) &dev_bools, n * sizeof(int));
            cudaMalloc((void**) &dev_indices, n * sizeof(int));
            cudaMalloc((void**) &dev_scatter, n * sizeof(int));
            cudaMalloc((void**) &dev_input, n * sizeof(int));
            cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int host_indices[n];
            int host_bools[n];

        	int blocksPerGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        	// 1. Create boolean array
            Common::kernMapToBoolean<<< blocksPerGrid, BLOCK_SIZE  >>>(n, dev_bools, dev_input);
            cudaMemcpy(host_bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
            // 2. Scan to generate indices
            scan(n, host_indices, host_bools);
            cudaMemcpy(dev_indices, host_indices, n * sizeof(int), cudaMemcpyHostToDevice);
            // 3. Scatter
            timer().startGpuTimer();
            Common::kernScatter<<< blocksPerGrid, BLOCK_SIZE >>>(n, dev_scatter, dev_input, dev_bools, dev_indices);
            timer().endGpuTimer();

            // Copy to output
            cudaMemcpy(odata, dev_scatter, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Memory cleanup
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_scatter);
            cudaFree(dev_input);

            // Beware! Since exclusive scan, we won't count last element in
            // indices, let's fix that
            if (host_bools[n - 1] != 0) {
            	return host_indices[n - 1] + 1;
            } else {
            	return host_indices[n-1];
            }
        }
    }
}
