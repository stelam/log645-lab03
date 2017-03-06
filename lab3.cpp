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
void dispatchTask(int m, int n, int nb_procs);
void processResult(double *matrix, int m, int n, int np);
void executeTask(int m, int n, double td, double h);
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
		printf("\n================================== Initiale ================================== \n");
		init_matrix(matrix, M, N, NP);
		printf("Matrice initiale : \n");
		print_matrix_at_k(matrix, M, N, NP, 0);
		printf("\n================================== Séquentiel ================================== \n");
		start_timer(&time_start);
		sequential(matrix, M, N, NP, TD, H);
		printf("Matrice finale : \n");
		print_matrix_at_k(matrix, M, N, NP, NP - 1);
		time_seq = stop_timer(&time_start);
		printf("\n================================== Parallel ================================== \n");
		init_matrix(matrix, M, N, NP);
		start_timer(&time_start);
		processResult(matrix, M, N, NP);
		printf("Matrice finale : \n");
		print_matrix_at_k(matrix, M, N, NP, NP - 1);
		time_parallel = stop_timer(&time_start);
		printf("================================================================================ \n");

		acc = time_seq/time_parallel;
		printf("Accéleration: %lf\n\n", acc );

	}
	else if(processor_rank == 1) {
		dispatchTask(M, N, NB_PROCS);

	}
	else {
		executeTask(M, N, TD, H);
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
				if(i != 0){ref2 = matrix[get_offset(k-1, i-1, j, m, n)];}
				else{ref2 = 0;}
				if(i != m-1){ref3 = matrix[get_offset(k-1, i+1, j, m, n)];}
				else{ref3 = 0;}
				if(j != 0){ref4 = matrix[get_offset(k-1, i, j-1, m, n)];}
				else{ref4 = 0;}
				if(j != n-1){ref5 = matrix[get_offset(k-1, i, j+1, m, n)];}
				else{ref5 = 0;}

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

void dispatchTask(int m, int n, int nb_procs) {
	int procRank = 2, tailleMatrix = m * n;
	double matrix[tailleMatrix+1], refs[8], continu = 1;
	
	while (continu == 1) {
		MPI_Recv(matrix, tailleMatrix+1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		continu = matrix[tailleMatrix];

		if (continu == 0)
		{
			refs[7] = 0;
			for (int p = 2; p < nb_procs; ++p)
			{
				MPI_Send(refs, 8, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
			
			}
			return;
		}

		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m; i++) {
				if (procRank == nb_procs){
					procRank = 2;
				}
				refs[0] = i;
				refs[1] = j;
				refs[2] = matrix[get_offset(0, i, j, m, n)];
				if(i != 0){refs[3] = matrix[get_offset(0, i-1, j, m, n)];}
				else{refs[3] = 0;}
				if(i != m-1){refs[4] = matrix[get_offset(0, i+1, j, m, n)];}
				else{refs[4] = 0;}
				if(j != 0){refs[5] = matrix[get_offset(0, i, j-1, m, n)];}
				else{refs[5] = 0;}
				if(j != n-1){refs[6] = matrix[get_offset(0, i, j+1, m, n)];}
				else{refs[6] = 0;}
				refs[7] = 1;

				MPI_Send(refs, 8, MPI_DOUBLE, procRank, 0, MPI_COMM_WORLD);
				procRank++;
			}
		}		
	}
}

void processResult(double *matrix, int m, int n, int np) {
	int tailleMatrix = m * n, ri, rj;
	double currentMatrix[tailleMatrix+1], result[3], res;

	for (int j = 0; j < n; j++) {
		for (int i = 0; i < m; i++) {
			currentMatrix[get_offset(0, i, j, m, n)] = matrix[get_offset(0, i, j, m, n)];
		}
	}
	currentMatrix[tailleMatrix] = 1;
	
	for (int k = 1; k < np; k++) {
		MPI_Send(currentMatrix, tailleMatrix+1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m; i++) {
				MPI_Recv(result, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				ri = (int)(result[0]);
				rj = (int)(result[1]);
				res = result[2];
				matrix[get_offset(k, ri, rj, m, n)] = res;
				currentMatrix[get_offset(0, ri, rj, m, n)] = res;
			}
		}
	}
	currentMatrix[tailleMatrix] = 0;
	MPI_Send(currentMatrix, tailleMatrix+1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
}

void executeTask(int m, int n, double td, double h) {
	double refs[8], ref1, ref2, ref3, ref4, ref5, result[3], continu = 1;
	int i, j; 

	while(continu == 1) {
		MPI_Recv(refs, 8, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		continu = refs[7];
		if (continu == 0)
		{
			return;
		}

		result[0] = refs[0];
		result[1] = refs[1];
		i = (int)refs[0];
		j = (int)refs[1];
		ref1 = refs[2];
		ref2 = refs[3];
		ref3 = refs[4];
		ref4 = refs[5];
		ref5 = refs[6];
		
		if (i == 0 || i == m -1 || j == 0 || j == n - 1) {
			result[2] = 0;
		}
		else {
			result[2] = ((1 - ((4 * td) / (h*h))) * ref1) + (td / (h*h)) * (ref2+ref3+ref4+ref5);
		}
		usleep(SLEEP_TIME);
		
		MPI_Send(result, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	
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
