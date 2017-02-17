#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include <mpi.h>
#include <sys/time.h>
#include <unistd.h>

#define MATRIX_ROW_LENGTH 8

void print_final_matrix(int *matrix);
void init_matrix(int *matrix, int matrixSize, int starting_value);
int get_offset(int k, int i, int j);
int first_traitement(int k, int pval, int i, int j);
int second_traitement(int k, int pval, int i, int j, int pj);
void initMPI(int *argc, char ***argv, int &number_of_processors, int &processor_rank);
void first_parallel_operation(int number_of_processors, int processor_rank, int *matrix, int k, int starting_value);
void second_parallel_operation(int number_of_processors, int processor_rank, int matrix[], int k, int starting_value);

int main(int argc, char** argv) {
	const int K = atoi(argv[3]); 
	const int MATRIX_SIZE = MATRIX_ROW_LENGTH * MATRIX_ROW_LENGTH;

	int number_of_processors, processor_rank;
	int starting_value = atoi(argv[2]);
	int matrix[MATRIX_SIZE];
	double timeStart, timeEnd, Texec;

	// chronomètre
	struct timeval tp;
	gettimeofday (&tp, NULL);
	timeStart = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
	
	init_matrix(matrix, MATRIX_SIZE, starting_value);
	initMPI(&argc, &argv, number_of_processors, processor_rank);

	if(atoi(argv[1]) == 1) {
		first_parallel_operation(number_of_processors, processor_rank, matrix, K, starting_value);
	}
	else {
		second_parallel_operation(number_of_processors, processor_rank, matrix, K, starting_value);
	}

	if(processor_rank == 0) {
		gettimeofday (&tp, NULL); // Fin du chronometre
		timeEnd = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
		Texec = timeEnd - timeStart; //Temps d'execution en secondes
		printf("Temps d execution: %lf\n", Texec);
	}

	MPI_Finalize();
}

/*
* Function: initMPI
* ----------------------------
*	Initializes MPI library and gets the number of available processors as well as the current processor's rank
*
*   argc: main's argc
*	argv: to main's argv
*	number_of_processors: pointer to the variable
*	number_of_processors: pointer to the variable
*
*/
void initMPI(int *argc, char ***argv, int &number_of_processors, int &processor_rank) {
	MPI_Init(&*argc, &*argv);
	MPI_Comm_size(MPI_COMM_WORLD, &number_of_processors);
	MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);
}

/*
* Function: get_offset
* ----------------------------
*   Transforms a 3D set of params into a 1D param (index) in order to get equivalent index in a 1D array.
*
*   k: the k index (iteration #)
*   i: the i param of a 2D array (x)
*   j: the j param of a 2D array (y)
*
*   returns: the 1D index equivalent (offset)
*/
int get_offset(int k, int i, int j) { 
	return (k * MATRIX_ROW_LENGTH * MATRIX_ROW_LENGTH) + (i * MATRIX_ROW_LENGTH) + j; 
}


/*
* Function: first_parallel_operation
* ----------------------------
*	Performs the first operation of the lab
*
*   number_of_processors: the number of available processors for the parallel processing
*   processor_rank: the rank of the current processor
*	matrix: the matrix that will contain the final results
*	k: the number of alterations to perform
*	starting_value: the initialization value for the initial matrix cells
*
*/
void first_parallel_operation(int number_of_processors, int processor_rank, int matrix[], int k, int starting_value) {

	int msg[2]; // 0 = matrix offset, 1 = computed value
	int received_msg = 0;

	// if proc 0 fails, everything fails
	if (processor_rank == 0) {
		while (received_msg < (number_of_processors - 1) ) {
			MPI_Recv(msg, 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			matrix[msg[0]] = msg[1];
			received_msg++;
		}
		print_final_matrix(matrix);
		
	} else {
		int i = processor_rank / MATRIX_ROW_LENGTH;
		int j = processor_rank % MATRIX_ROW_LENGTH;
		msg[0] = processor_rank;

		int final_value = starting_value;
		for (int current_k = 1; current_k <= k; current_k++) {
			usleep(1000);
			final_value = final_value + (i+j) * current_k;
		}

		// send the final results to the master
		msg[1] = final_value;
		MPI_Send(msg, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
}


/*
* Function: second_parallel_operation
* ----------------------------
*	Performs the second operation of the lab
*
*   number_of_processors: the number of available processors for the parallel processing
*   processor_rank: the rank of the current processor
*	matrix: the matrix that will contain the final results
*	k: the number of alterations to perform
*	starting_value: the initialization value for the initial matrix cells
*
*/
void second_parallel_operation(int number_of_processors, int processor_rank, int matrix[], int k, int starting_value) {

	int msg[2]; // 0 = matrix offset, 1 = computed value
	int received_msg = 0;

	// if proc 0 fails, everything fails
	if (processor_rank == 0) {
		while (received_msg < (number_of_processors - 1) ) {
			MPI_Recv(msg, 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			matrix[msg[0]] = msg[1];
			received_msg++;
		}
		print_final_matrix(matrix);
		
	} else {
		int p_i = processor_rank / MATRIX_ROW_LENGTH;
		int p_j = processor_rank % MATRIX_ROW_LENGTH;
		int previousProcessor = processor_rank-1;
		int nextProcessor = processor_rank+1;
		msg[0] = processor_rank;

		int previous_k_value = starting_value;
		int previous_j_value = starting_value;


		// for each k
		for (int current_k = 1; current_k <= k; current_k++) {

			if(p_j == 0)
			{
				previous_k_value = previous_k_value + (p_i * current_k);
				usleep(1000);
				MPI_Send(&previous_k_value, 1, MPI_INT, nextProcessor, 0, MPI_COMM_WORLD);
			}
			else
			{
				//processor 0 never sends any message, so previous j equals to the starting value for processor 1
				if(processor_rank != 1) {
					MPI_Recv(&previous_j_value, 1, MPI_INT, previousProcessor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}

				previous_k_value = previous_k_value + (previous_j_value * current_k);
				usleep(1000);
				
				if (p_j != 7)
				{	
					MPI_Send(&previous_k_value, 1, MPI_INT, nextProcessor, 0, MPI_COMM_WORLD);
				}
			}
			
		}

		// send the last value calculated to the master
		msg[1] = previous_k_value;
		MPI_Send(msg, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
}

/*
* Function: init_matrix
* ----------------------------
*	Initializes each index of an array to a given value
*
*   matrix: the 1D array containing the matrixes
*   matrixSize: the size of one matrix (contained in the array)
*	starting_value: the value that will be used to set the indexes
*
*/
void init_matrix(int *matrix, int matrixSize, int startingValue) {
	int i;
	for (i = 0; i < matrixSize; i++) {
		matrix[i] = startingValue;
	}
}


/*
* Function: print_final_matrix
* ----------------------------
*	Prints the last matrix that is the result of a series of operations
*
*   matrix: the 1D array containing the matrix
*
*/
void print_final_matrix(int *matrix) {
	for (int i = 0; i < MATRIX_ROW_LENGTH; i++){
		for (int j = 0; j < MATRIX_ROW_LENGTH; j++) {
			printf("%7d \t", matrix[get_offset(0, i, j)]);
		}
		printf("\n");
	}
	printf("\n\n");
}
