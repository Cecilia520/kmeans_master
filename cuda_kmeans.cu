#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "kmeans.h"

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number){
	if(err!=cudaSuccess){
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

#define STREAMS_CNT 4

static inline int nextPowerOfTwo(int n) {
	n--;

	n = n >>  1 | n;
	n = n >>  2 | n;
	n = n >>  4 | n;
	n = n >>  8 | n;
	n = n >> 16 | n;
	//  n = n >> 32 | n;    //  For 64-bit ints

	return ++n;
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static
float euclid_dist_2(int    numvariables,
		int    numObjs,
		int    numClusters,
		float *objects,     // [numvariables][numObjs]
		float *clusters,    // [numvariables][numClusters]
		int    objectId,
		int    clusterId) {
	float ans=0.0;
	for (int i = 0; i < numvariables; i++) {
		ans +=(objects[numObjs*i+objectId] - clusters[numClusters * i + clusterId])*(objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
	}

	return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static void find_nearest_cluster(int numvariables,
		int numObjs,
		int numClusters,
		float *objects,           //  [numvariables][numObjs]
		float *deviceClusters,    //  [numvariables][numClusters]
		int *membership,          //  [numObjs]
		int *intermediates){

	//  The type chosen for membershipChanged must be large enough to support reductions!
	//  There are blockDim.x elements, one for each thread in the block.
	//  See numThreadsPerClusterBlock in cuda_kmeans().
	//unsigned char *membershipChanged = (unsigned char *)sharedMemory;

	__shared__ unsigned char membershipChanged[128];

	float *clusters = deviceClusters;
	membershipChanged[threadIdx.x] = 0;

	int objectId = blockDim.x * blockIdx.x + threadIdx.x;
	if (objectId < numObjs) {
		int   index, i;
		float dist, min_dist;

		/* find the cluster id that has min distance to object */
		index    = 0;
		min_dist = euclid_dist_2(numvariables, numObjs, numClusters, objects, clusters, objectId, 0);
		for (i=1; i<numClusters; i++) {
			dist = euclid_dist_2(numvariables, numObjs, numClusters, objects, clusters, objectId, i);
			/* no need square root */
			if (dist < min_dist) { /* find the min and its array index */
				min_dist = dist;
				index    = i;
			}
		}

		if (membership[objectId] != index) {
			membershipChanged[threadIdx.x] = 1;
		}

		/* assign the membership to object objectId */
		membership[objectId] = index;

		__syncthreads();    //  For membershipChanged[]

		// blockDim.x *must* be a power of two!
		// this is a reduction
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (threadIdx.x < s) {
				membershipChanged[threadIdx.x] += membershipChanged[threadIdx.x + s];
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			intermediates[blockIdx.x] = membershipChanged[0];
		}
	}
}

__global__ static void compute_delta(int *deviceIntermediates,
		int numIntermediates,    //  The actual number of intermediates
		int numIntermediates2)   //  The next power of two
{
	//  The number of elements in this array should be equal to numIntermediates2, the number of threads launched.
	//  It *must* be a power of two!
	extern __shared__ unsigned int intermediates[];

	//  Copy global intermediate values into shared memory.
	intermediates[threadIdx.x] = (threadIdx.x < numIntermediates) ? deviceIntermediates[threadIdx.x] : 0;

	__syncthreads();

	//  numIntermediates2 *must* be a power of two!
	for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		deviceIntermediates[0] = intermediates[0];
	}
}

/*----< cuda_kmeans() >-------------------------------------------------------*/
//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numvariables]
//  clusters        [numClusters][numvariables]
//  dimObjects      [numvariables][numObjs]
//  dimClusters     [numvariables][numClusters]
//  newClusters     [numvariables][numClusters]
//  deviceObjects   [numvariables][numObjs]
//  deviceClusters  [numvariables][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numvariables]       */
float** cuda_kmeans(float **objects,      /* in: [numObjs][numvariables] */
		int     numvariables,    /* no. features */
		int     numObjs,      /* no. objects */
		int     numClusters,  /* no. clusters */
		float   threshold,    /* % objects change membership */
		int    *membership,   /* out: [numObjs] */
		int    *loop_iterations)
{
	int      i, j, index, loop=0;
	int     *newClusterSize; /* [numClusters]: no. objects assigned in each
								new cluster */
	float    delta;          /* % of objects change their clusters */
	float	**dimObjects;
	float	**clusters;       /* out: [numClusters][numvariables] */
	float	**dimClusters;
	float	**newClusters;    /* [numvariables][numClusters] */

	float	*deviceObjects;
	float	*deviceClusters;
	int		*deviceMembership;
	int		*deviceIntermediates;

	//  Copy objects given in [numObjs][numvariables] layout to new
	//  [numvariables][numObjs] layout
	malloc2D(dimObjects, numvariables, numObjs, float);
	for (i = 0; i < numvariables; i++) {
		for (j = 0; j < numObjs; j++) {
			dimObjects[i][j] = objects[j][i];
		}
	}

	/* pick first numClusters elements of objects[] as initial cluster centers*/
	malloc2D(dimClusters, numvariables, numClusters, float);
	for (i = 0; i < numvariables; i++) {
		for (j = 0; j < numClusters; j++) {
			dimClusters[i][j] = dimObjects[i][j];
		}
	}

	/* initialize membership[] */
	for (i=0; i<numObjs; i++)
		membership[i] = -1;

	/* need to initialize newClusterSize and newClusters[0] to all 0 */
	newClusterSize = (int*)calloc(numClusters, sizeof(int));
	assert(newClusterSize != NULL);

	malloc2D(newClusters, numvariables, numClusters, float);
	memset(newClusters[0], 0, numvariables * numClusters * sizeof(float));

	// To support reduction, numThreadsPerClusterBlock *must* be a power of two,
	// and it *must* be no larger than the number of bits that will fit into an unsigned char,
	// the type used to keep track of membership changes in the kernel.
	const unsigned int numThreadsPerClusterBlock = 128;
	const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
	//const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock * sizeof(unsigned char);
	const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);
	const unsigned int reductionBlockSharedDataSize = numReductionThreads * sizeof(unsigned int);

	SAFE_CALL((cudaMalloc(&deviceObjects, numObjs*numvariables*sizeof(float))), "CUDA malloc error!");
	SAFE_CALL((cudaMalloc(&deviceClusters, numClusters*numvariables*sizeof(float))), "CUDA malloc error!");
	SAFE_CALL((cudaMalloc(&deviceMembership, numObjs*sizeof(int))), "CUDA malloc error!");
	SAFE_CALL((cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int))), "CUDA malloc error!");

	SAFE_CALL(cudaMemcpy(deviceObjects, dimObjects[0], numObjs*numvariables*sizeof(float), cudaMemcpyHostToDevice), "CUDA memory release error!");
	SAFE_CALL(cudaMemcpy(deviceMembership, membership, numObjs*sizeof(int), cudaMemcpyHostToDevice), "CUDA memory release error!");

	do{
		SAFE_CALL((cudaMemcpy(deviceClusters, dimClusters[0], numClusters*numvariables*sizeof(float), cudaMemcpyHostToDevice)),
				"CUDA memory copy from host to device error!");

		find_nearest_cluster<<<numClusterBlocks, numThreadsPerClusterBlock>>>(numvariables,
				numObjs,
				numClusters,
				deviceObjects,
				deviceClusters,
				deviceMembership,
				deviceIntermediates);


		cudaDeviceSynchronize();

		compute_delta<<<1,numReductionThreads, reductionBlockSharedDataSize >>>(deviceIntermediates, numClusterBlocks, numReductionThreads);
		cudaDeviceSynchronize();

		int d;
		SAFE_CALL((cudaMemcpy(&d, deviceIntermediates, sizeof(int), cudaMemcpyDeviceToHost)), "CUDA memory copy from device to host error!");
		delta = (float)d;

		SAFE_CALL((cudaMemcpy(membership, deviceMembership, numObjs*sizeof(int), cudaMemcpyDeviceToHost)), "CUDA memory copy from device to host error!");

		for (i=0; i<numObjs; i++) {
			/* find the array index of nestest cluster center */
			index = membership[i];

			/* update new cluster centers : sum of objects located within */
			newClusterSize[index]++;
			for (j=0; j<numvariables; j++)
				newClusters[j][index] += objects[i][j];
		}

		//  TODO: Flip the nesting order
		//  TODO: Change layout of newClusters to [numClusters][numvariables]
		// average the sum and replace old cluster centers with newClusters
		for (i=0; i<numClusters; i++) {
			for (j=0; j<numvariables; j++) {
				if (newClusterSize[i] > 0)
					dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
				newClusters[j][i] = 0.0;   /* set back to 0 */
			}
			newClusterSize[i] = 0;   /* set back to 0 */
		}
		delta /= numObjs;
	} while (delta > threshold && loop++ < 500);

	*loop_iterations = loop + 1;

	/* allocate a 2D space for returning variable clusters[] (coordinates of cluster centers) */
	malloc2D(clusters, numClusters, numvariables, float);
	for (i = 0; i < numClusters; i++) {
		for (j = 0; j < numvariables; j++) {
			clusters[i][j] = dimClusters[j][i];
		}
	}

	SAFE_CALL((cudaFree(deviceObjects)), "CUDA memory release error!");
	SAFE_CALL((cudaFree(deviceClusters)), "CUDA memory release error!");
	SAFE_CALL((cudaFree(deviceMembership)),"CUDA memory release error!");
	SAFE_CALL((cudaFree(deviceIntermediates)),"CUDA memory release error!");

	free(dimObjects[0]);
	free(dimObjects);
	free(dimClusters[0]);
	free(dimClusters);
	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);

	return clusters;
}
