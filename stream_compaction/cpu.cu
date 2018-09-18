#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        scanNoTimer(n, odata, idata);
	        timer().endCpuTimer();
        }

        void scanNoTimer(int n, int *odata, const int *idata) {
        	odata[0] = 0; //Add identity
        	int sum = 0;
        	for (int i = 0; i < n - 1; i++){
        		sum += idata[i];
        		odata[i + 1] = sum;
        	}
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            int index = 0;

	        for (int i = 0; i < n; i++) {
            	if (idata[i] != 0){
            		odata[index] = idata[i];
            		index ++;
            	}
            }
	        timer().endCpuTimer();
            return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
        	int idx = 0;
        	int booleanArray[n];
        	timer().startCpuTimer();
        	for (int i = 0; i < n; i++) {
        		if (idata[i] != 0) {
        			booleanArray[i] = 1;
        		} else {
        			booleanArray[i] = 0;
        		}
        	}
        	int indexArray[n];
        	scanNoTimer(n, indexArray, booleanArray);
        	// Finally, scatter
        	for (int i = 0; i < n; i++) {
        		if (idata[i] != 0) {
        			idx = indexArray[i];
        			odata[idx] = idata[i];
        		}
        	}
	        timer().endCpuTimer();
            return idx + 1;
        }
    }
}
