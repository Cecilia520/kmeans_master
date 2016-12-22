#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>   /*gettimeofday()*/
#include <omp.h>
#include <stdint.h>
// self-defined header file
#include "kmeans.h"

int main(int argc, char *argv[]) {
    int     i, j;
    int    *prediction;    /* [numObs] */
    float **dat;           /* [numObs][numvariables] data dat */
    float **clusters;      /* [numClusters][numvariables] cluster center */
    int     loop_iterations;

    struct timeval start, end;

    int numvariables;
    int numObs;

    // set some default values
    int nthreads         = 0;
    int numClusters      = 0;
    float threshold      = 0.001;
    char  *filename      = NULL;

    // assign the input arguments to variables
    filename = argv[1];
    numClusters = atoi(argv[2]); // change variable type to int
    threshold = atof(argv[3]);   // change variable type to float
    nthreads = atoi(argv[4]);

    // check arguments conditions

    if (filename == NULL || numClusters <= 1 || threshold <= 0.000 || nthreads <= 1)
            {printf("Error Input!! Please read README file!\n");exit(0);}
    /* set the number of threads if specified in command line, else retrun error message*/
    if (nthreads > 0) omp_set_num_threads(nthreads);

    /* read data points from file ------------------------------------------*/
    dat = file_read(filename, &numObs, &numvariables);

    // record the starting time
    gettimeofday(&start,NULL);  /*clock_gettime is not good for mac OX system */

    /* start the core computation -------------------------------------------*/
    /* prediction: the cluster id for each data object */
    prediction = (int*) malloc(numObs * sizeof(int));

    clusters = omp_kmeans(dat, numvariables, numObs, numClusters, threshold,
                            prediction, &loop_iterations);

    free(dat[0]);
    free(dat);

    // record the endting time
    gettimeofday(&end,NULL);    /*clock_gettime is not good for mac OX system */
    uint64_t clustering_timing = 1000000L * (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec);

    /* output: the coordinates of the cluster centres ----------------------*/
    file_write(filename, numClusters, numObs, numvariables, clusters, prediction);

    free(prediction);
    free(clusters[0]);
    free(clusters);

    /*---- output performance numbers ---------------------------------------*/
    printf("\n================ Regular Kmeans (OpenMP) ================\n");
    printf("numObs              = %d\n", numObs);
    printf("numvariables        = %d\n", numvariables);
    printf("numClusters         = %d\n", numClusters);
    printf("threshold           = %.4f\n", threshold);
    printf("nthreads            = %d\n", nthreads);
    printf("Loop iterations     = %d\n", loop_iterations);
    printf("Computation timing  = %lu microseconds\n", (long unsigned int) clustering_timing);

    return(0);
}
