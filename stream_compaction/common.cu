#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= n) { return; }

            if (idata[i] != 0) {
            	bools[i] = 1;
            } else {
            	bools[i] = 0;
            }
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;

            int outIndex = -1;
            if (bools[i] == 1) {
            	outIndex = indices[i];
            	odata[outIndex] = idata[i];
            }
        }

        __global__ void kernConvertScanToExclusive(int n, int exclusiveScan[], const int inclusiveScan[]) {
        	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
        	if (idx >= n) {
        		return;
        	} else if (idx >= 1) {
        		exclusiveScan[idx] = inclusiveScan[idx - 1];
        		return;
        	}

        	exclusiveScan[0] = 0;
        }

    }
}
