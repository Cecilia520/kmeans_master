#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "kmeans.h"

/* square of Euclid distance between two multi-dimensional points            */
float distance(int numdims, float *X1, float *X2)
/* no. dimensions */
/* [numdims] */
/* [numdims] */
{
    int i;
    float ans=0.0;
    for (i=0; i<numdims; i++)
        ans += (X1[i]-X2[i]) * (X1[i]-X2[i]);

    return(ans);
}

int clusterdecision(int numClusters, int numvariables, float  *object, float **clusters)
/* no. clusters */
/* no. coordinates */
/* [numvariables] */
/* [numClusters][numvariables] */
{
    int   index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = distance(numvariables, object, clusters[0]);
    for (i=1; i<numClusters; i++) {
        dist = distance(numvariables, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}


/* return an array of cluster centers of size [numClusters][numvariables] */
float** omp_kmeans(float **dat, int numvariables, int numObs, int numClusters,float threshold, int *prediction,int *loop_iterations)
/* in: [numObs][numvariables] */
/* no. features */
/* no. dat */
/* no. clusters */
/* % dat change prediction */
/* out: [numObs] */
{
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. dat assigned in each
                                new cluster */
    float    delta;          /* % of dat change their clusters */
    float  **clusters;       /* out: [numClusters][numvariables] */
    float  **newClusters;    /* [numClusters][numvariables] */

    /* allocate a 2D space for returning variable clusters[] (variables
       of cluster centers) */
    clusters    = (float**) malloc(numClusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(numClusters * numvariables * sizeof(float));
    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + numvariables;       //Question needs to be confirmed

    /* pick first numClusters elements of dat[] as initial cluster centers*/
    for (i=0; i<numClusters; i++)
        for (j=0; j<numvariables; j++)
            clusters[i][j] = dat[i][j];
    // printf("%d\n",i);      // 4
    // printf("%d\n",j);      // 63

    /* initialize prediction[] by setting each prediction result -1*/
    for (i=0; i<numObs; i++) prediction[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));

    /*
    The malloc only allocate the memory with single arguments;
    The calloc do allocate the memory with double arguments and initialize the value to 0;
    */

    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    newClusters[0] = (float*)  calloc(numClusters * numvariables, sizeof(float));
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numvariables;

    do {
        delta = 0.0;
        #pragma omp parallel for private(i,j,index)
        for (i=0; i<numObs; i++)
        {
            /* find the array index of nestest cluster center */
            index = clusterdecision(numClusters, numvariables, dat[i], clusters);

            /* if prediction changes, increase delta by 1 */
            if (prediction[i] != index) delta += 1.0;

            /* assign the prediction to object i */
            prediction[i] = index;

            /* update new cluster centers : sum of dat located within */
            newClusterSize[index]++;
            for (j=0; j<numvariables; j++)
                newClusters[index][j] += dat[i][j];
        }

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++)
        {
            for (j=0; j<numvariables; j++)
            {
                if (newClusterSize[i] > 0)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }

        delta /= numObs;
    } while (delta > threshold && loop++ < 500); // stop untill the results getting converge
    *loop_iterations = loop + 1;
    /* we want to know how many loops we need to perform in order to get the final
    clustered prediction class in the output*/

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}
