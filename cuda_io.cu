#include <stdio.h>
#include <stdlib.h>
#include <string.h>     //strtok(): breaking a string into a series of tokens
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */

#include "kmeans.h"
#define MAXNUM_PERLINE 128

////////////////////////////////////////////////////////////////
/*Read in file into the critical format for cluster analysis*///
////////////////////////////////////////////////////////////////

float** file_read(char *filename, int *numObs, int *numvariables)
/*
 argument instruction
 filename: input file name
 numObs: number of data observations (local)
 numvariables: number of the data features (variables)
*/
{
    float **dat;
    int     i, j, len;

    FILE *infile;
    char *line, *ret;
	//char *line;
    int   lineLen;

    // open the data file to prepare for read in
    infile = fopen(filename, "r");

    // first find the number of objects
    lineLen = MAXNUM_PERLINE;
    line = (char*) malloc(lineLen);

    (*numObs) = 0;
    while (fgets(line, lineLen, infile) != NULL) {
        /* check each line to find the max line length */
        while (strlen(line) == lineLen-1) {
            /* this line read is not complete */
            len = strlen(line);
            fseek(infile, -len, SEEK_CUR);
            /* increase lineLen */
            lineLen += MAXNUM_PERLINE;
            line = (char*) realloc(line, lineLen);

            ret = fgets(line, lineLen, infile);
        }

        if (strtok(line, " \t\n") != 0) // check the number of observations when there is a "\n"
            (*numObs)++;
    }
    rewind(infile);
    printf("lineLen = %d\n",lineLen);

    // find the number of variables
    (*numvariables) = 0;
    while (fgets(line, lineLen, infile) != NULL) {
        if (strtok(line, " \t\n") != 0) {
            /* ignore the id (first coordiinate): numvariables = 1; */
            while (strtok(NULL, " ,\t\n") != NULL) (*numvariables)++;
            break; /* this makes read from 1st object */
        }
    }
    rewind(infile);

    printf("File (%s) has %d number of Observations.\n",filename,*numObs);
    printf("File (%s) has %d number of Variables.\n",filename,*numvariables);

    /* allocate space for objects[][] and read all objects */
    len = (*numObs) * (*numvariables);
    dat    = (float**)malloc((*numObs) * sizeof(float*));
    dat[0] = (float*) malloc(len * sizeof(float));
    for (i=1; i<(*numObs); i++)
        dat[i] = dat[i-1] + (*numvariables);

    i=0;
    /* read all objects */
    while (fgets(line, lineLen, infile) != NULL) {
        if (strtok(line, " \t\n") == NULL) continue;
        for (j=0; j<(*numvariables); j++)
            dat[i][j] = atof(strtok(NULL, " ,\t\n"));
        i++;
    }

    fclose(infile);
    free(line);

    return dat;
}

////////////////////////////////////////////////////////////////
///////*Write the Results from calculation to text file*////////
////////////////////////////////////////////////////////////////
int file_write(char *filename, int numClusters, int numObs, int numvariables, float **clusters, int *prediction)
/* dimension information in the behind*/
/* input file name */
/* no. clusters */
/* no. data objects */
/* no. variables (local) */
/* [numClusters][numvariables] centers */
/* [numObs] */
{
    FILE *fptr;
    int   i, j;
    char  outFileName[1024];

    /* output: the variables of the cluster centres ----------------------*/
    sprintf(outFileName, "%s.cluster_centres", filename);
    printf("Writing variables of K=%d cluster centers to file \"%s\"\n",
           numClusters, outFileName);
    fptr = fopen(outFileName, "w");
    for (i=0; i<numClusters; i++) {
        fprintf(fptr, "%d ", i);
        for (j=0; j<numvariables; j++)
            fprintf(fptr, "%f ", clusters[i][j]);
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    /* output: the closest cluster centre to each of the data points --------*/
    sprintf(outFileName, "%s.prediction", filename);
    printf("Writing prediction of N=%d data objects to file \"%s\"\n",
           numObs, outFileName);
    fptr = fopen(outFileName, "w");
    for (i=0; i<numObs; i++)
        fprintf(fptr, "%d %d\n", i, prediction[i]);
    fclose(fptr);

    return 1;
}
