#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
// self-defined header file
#include <mpi.h> // included for the mpi function
#include "kmeans.h"

// what the real meaning of this part ????????????????
int     mpi_kmeans(float**, int, int, int, float, int*, float**, MPI_Comm);
float** mpi_read(char*, int*, int*, MPI_Comm);
int     mpi_write(char*, int, int, int, float**, int*, int, MPI_Comm);
// what the real meaning of this part ????????????????

int main(int argc, char *argv[]) {
           //int     opt;
    //extern char   *optarg;
    //extern int     optind;

    int i,j;

    // two dimensional dataset information
    int    *prediction;    /* [numObs] */
    float **dat;       /* [numObs][numvariables] data dat */
    float **clusters;      /* [numClusters][numvariables] cluster center */
    // no loop_iterations since different iterations in each processes

    int numvariables;
    int numObs;
    int totalnumObs;

    // We need to consider the time for broadcast the dat file and the computation time.
    // initialize the time measurement variables
    double time_preend, time_prestart, time_transferdata;
    double time_start, time_end, clustering_timing;
    // MPI related varaibles
    int rank, nproc, mpi_namelen;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    /* some default values */
    int numClusters      = 0;
    float threshold      = 0.001;
    char  *filename      = NULL;

    // assign the input arguments to variables
    filename = argv[1];
    numClusters = atoi(argv[2]); // change variable type to int
    threshold = atof(argv[3]);   // change variable type to float

    if (filename == NULL || numClusters <= 1 || threshold <= 0.000)
        {   printf("Error Input!! Please read README file!\n");
            if (rank == 0)
                {printf("Error Input on master process! Please read README file!\n");}
            MPI_Finalize(); // finalize MPI if incorrect input
            exit(1);
        }

    MPI_Barrier(MPI_COMM_WORLD);  // waiting for all MPI processes ready

    time_prestart = MPI_Wtime();
    /* read data points from file ------------------------------------------*/
    dat = mpi_read(filename, &numObs, &numvariables, MPI_COMM_WORLD);

    time_preend = MPI_Wtime();
    // time for transfer data
    time_transferdata = time_preend - time_prestart;

    // computation start time
    time_start = MPI_Wtime();

    /* allocate a 2D space for clusters[] (coordinates of cluster centers)
       this array should be the same across all processes                  */
    clusters    = (float**) malloc(numClusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(numClusters * numvariables * sizeof(float));

    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + numvariables;

    MPI_Allreduce(&numObs, &totalnumObs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    /* pick first numClusters elements in feature[] as initial cluster centers*/
    if (rank == 0) {
        for (i=0; i<numClusters; i++)
            for (j=0; j<numvariables; j++)
                clusters[i][j] = dat[i][j];
    }
    MPI_Bcast(clusters[0], numClusters*numvariables, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* prediction: the cluster id for each data object */
    prediction = (int*) malloc(numObs * sizeof(int));
    assert(prediction != NULL);

    /* start the core computation -------------------------------------------*/
    mpi_kmeans(dat, numvariables, numObs, numClusters, threshold, prediction,
               clusters, MPI_COMM_WORLD);

    free(dat[0]);
    free(dat);

    time_end = MPI_Wtime(); // computation start time
    clustering_timing = time_end - time_start; // computation time interval

    /* output: the coordinates of the cluster centres ----------------------*/
    mpi_write(filename, numClusters, numObs, numvariables, clusters, prediction, totalnumObs, MPI_COMM_WORLD);

    free(prediction);
    free(clusters[0]);
    free(clusters);

    /*---- output performance numbers ---------------------------------------*/
    double max_clustering_timing;

    /* get the max timing measured among all processes */
    MPI_Reduce(&clustering_timing, &max_clustering_timing, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nPerforming **** Simple Kmeans  (MPI) ****\n");
        printf("numObs                  = %d\n", totalnumObs);
        printf("numvariables            = %d\n", numvariables);
        printf("numClusters             = %d\n", numClusters);
        printf("threshold               = %.4f\n", threshold);
        printf("Num of processes        = %d\n", nproc);  // Only one processes here.
        printf("File Brocasting time    = %f sec\n", time_transferdata);
        printf("Computation time        = %f sec\n", max_clustering_timing);
    }

    MPI_Finalize();
    return(0);
}
