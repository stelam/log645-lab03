#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include <mpi.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstddef>

#define SLEEP_TIME 5

void initMPI(int *argc, char ***argv, int &number_of_processors, int &processor_rank);
void print_matrix_at_k(double *matrix, int m, int n, int np, int k);
void init_matrix(double *matrix, int m, int n, int np);
int get_offset(int k, int i, int j, int m, int n);
void first_parallel_operation(int number_of_processors, int processor_rank, int *matrix, int k, int starting_value);
void second_parallel_operation(int number_of_processors, int processor_rank, int matrix[], int k, int starting_value);
void sequential(double *matrix, int m, int n, int np, double td, double h);
void start_timer(double *time_start);
double stop_timer(double *time_start);

int main(int argc, char** argv) {
	const int N = atoi(argv[1]);
	const int M = atoi(argv[2]);
	const int NP = atoi(argv[3]);
	const double TD = atof(argv[4]);
	const double H = atof(argv[5]);
	const int NB_PROCS = atoi(argv[6]);

	double matrix[M * N * NP];
	int number_of_processors, processor_rank;
	double time_seq, time_parallel, acc, time_start;


	initMPI(&argc, &argv, number_of_processors, processor_rank);

	if(processor_rank == 0) {
		// sequential
		printf("\n================================== Séquentiel ================================== \n");
		init_matrix(matrix, M, N, NP);
		printf("Matrice initiale : \n");
		print_matrix_at_k(matrix, M, N, NP, 0);
		start_timer(&time_start);
		sequential(matrix, M, N, NP, TD, H);
		printf("Matrice finale : \n");
		print_matrix_at_k(matrix, M, N, NP, NP - 1);
		time_seq = stop_timer(&time_start);
		printf("================================================================================ \n\n\n");

		acc = time_seq/time_parallel;
		printf("Accéleration: %lf\n\n", acc );

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
*   i: the i param of a 2D array (m on x axis)
*   j: the j param of a 2D array (n on y axis)
*
*   returns: the 1D index equivalent (offset)
*/
int get_offset(int k, int i, int j, int m, int n) { 
	return (k * m * n) + (j * m) + i; 
}


void start_timer(double *time_start) {
	struct timeval tp;
	gettimeofday (&tp, NULL); // Debut du chronometre
	*time_start = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
}

double stop_timer(double *time_start) {
	struct timeval tp;
	double timeEnd, Texec;
	gettimeofday (&tp, NULL); // Fin du chronometre
	timeEnd = (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
	Texec = timeEnd - *time_start; //Temps d'execution en secondes
	printf("Temps d\'éxecution: %lf\n", Texec);
	return Texec;
}


void sequential(double *matrix, int m, int n, int np, double td, double h) {

	double ref1, ref2, ref3, ref4, ref5;

	//process
	for (int k = 1; k < np; k++) {
		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m; i++) {
				ref1 = matrix[get_offset(k-1, i, j, m, n)];
				ref2 = matrix[get_offset(k-1, i-1, j, m, n)];
				ref3 = matrix[get_offset(k-1, i+1, j, m, n)];
				ref4 = matrix[get_offset(k-1, i, j-1, m, n)];
				ref5 = matrix[get_offset(k-1, i, j+1, m, n)];

				if (i == 0 || i == m -1 || j == 0 || j == n - 1) {
					matrix[get_offset(k, i, j, m, n)] = 0;
				} else {
					matrix[get_offset(k, i, j, m, n)] = ((1 - ((4 * td) / (h*h))) * ref1) + (td / (h*h)) * (ref2+ref3+ref4+ref5);
				}

				usleep(SLEEP_TIME);

			}
		}		
	}
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

/*	int msg[2]; // 0 = matrix offset, 1 = computed value
	int received_msg = 0;

	// if proc 0 fails, everything fails
	if (processor_rank == 0) {
		while (received_msg < (number_of_processors - 1) ) {
			MPI_Recv(msg, 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			matrix[msg[0]] = msg[1];
			received_msg++;
		}
		//print_matrix_at_k(matrix);
		
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
	}*/
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

/*	int msg[2]; // 0 = matrix offset, 1 = computed value
	int received_msg = 0;

	// if proc 0 fails, everything fails
	if (processor_rank == 0) {
		while (received_msg < (number_of_processors - 1) ) {
			MPI_Recv(msg, 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			matrix[msg[0]] = msg[1];
			received_msg++;
		}
		//print_matrix_at_k(matrix);
		
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
	}*/
}

/*
* Function: init_matrix
* ----------------------------
*	Initializes each index of an array to a given value
*
*   matrix: the 1D array containing the matrixes
*   m: size on x axis
*	n: size on y axis
*
*/
void init_matrix(double *matrix, int m, int n, int np) {
	for (int k = 0; k < np; k++) {
		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m; i++) {
				if (k == 0)
					matrix[get_offset(0, i, j, m, n)] = i * (m - i - 1) * j * (n - j - 1);
				else {
					matrix[get_offset(k, i, j, m, n)] = -1;
				}
			}
		}
	}
}


/*
* Function: print_matrix_at_k
* ----------------------------
*	Prints the last matrix that is the result of a series of operations
*
*   matrix: the 1D array containing the matrix
*
*/
void print_matrix_at_k(double *matrix, int m, int n, int np, int k) {
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < m; i++){
			printf("%6.1f \t", matrix[get_offset(k, i, j, m, n)]);
		}
		printf("\n");
	}
	printf("\n\n");
}
