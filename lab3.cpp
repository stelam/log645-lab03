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
void print_matrix_at_k(double *matrix, int m, int n);
void init_matrix(double *matrix, int m, int n);
int get_offset(int i, int j, int m, int n);
void dispatchTask(int m, int n, int nb_procs);
void processResult(double *matrix, int m, int n, int np);
void executeTask(int m, int n, double td, double h);
void sequential(double *matrix, int m, int n, int np, double td, double h);
void start_timer(double *time_start);
double stop_timer(double *time_start);
void copyMatrix(double *matrixSource, double *matrixDest, int n, int m);

int main(int argc, char** argv) {
	const int N = atoi(argv[1]);
	const int M = atoi(argv[2]);
	const int NP = atoi(argv[3]);
	const double TD = atof(argv[4]);
	const double H = atof(argv[5]);
	const int NB_PROCS = atoi(argv[6]);

	double matrix[M * N];
	int number_of_processors, processor_rank;
	double time_seq, time_parallel, acc, time_start;


	initMPI(&argc, &argv, number_of_processors, processor_rank);

	int nbCaseCalc = ((M*N) - (2*M + 2*N -4) + 2);

	if(processor_rank == 0) {
		printf("\n================================== Initiale ================================== \n");
		init_matrix(matrix, M, N);
		printf("Matrice initiale : \n");
		print_matrix_at_k(matrix, M, N);
		printf("\n================================== Séquentiel ================================== \n");
		start_timer(&time_start);
		sequential(matrix, M, N, NP, TD, H);
		time_seq = stop_timer(&time_start);
		printf("Matrice finale : \n");
		print_matrix_at_k(matrix, M, N);
		printf("\n================================== Parallel ================================== \n");
		init_matrix(matrix, M, N);
		start_timer(&time_start);
		processResult(matrix, M, N, NP);
		time_parallel = stop_timer(&time_start);
		printf("Matrice finale : \n");
		print_matrix_at_k(matrix, M, N) ;
		printf("================================================================================ \n");

		acc = time_seq/time_parallel;
		printf("Accéleration: %lf\n\n", acc );

	}
	else if(processor_rank == 1) {
		(NB_PROCS < nbCaseCalc)?dispatchTask(M, N, NB_PROCS):dispatchTask(M, N, nbCaseCalc);		

	}
	else if(processor_rank < NB_PROCS && processor_rank < nbCaseCalc){
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
int get_offset(int i, int j, int m, int n) { 
	return (j * m) + i; 
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

	double ref1, ref2, ref3, ref4, ref5, matrixCurrent[m * n];

	//process
	for (int k = 1; k < np; k++) {

	copyMatrix(matrix, matrixCurrent, n, m);

		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m; i++) {
				if (i == 0 || i == m -1 || j == 0 || j == n - 1) {
					matrix[get_offset(i, j, m, n)] = 0;
				} else {
					ref1 = matrixCurrent[get_offset(i, j, m, n)];
					ref2 = matrixCurrent[get_offset(i-1, j, m, n)];
					ref3 = matrixCurrent[get_offset(i+1, j, m, n)];
					ref4 = matrixCurrent[get_offset(i, j-1, m, n)];
					ref5 = matrixCurrent[get_offset(i, j+1, m, n)];
									
					matrix[get_offset(i, j, m, n)] = ((1 - ((4 * td) / (h*h))) * ref1) + (td / (h*h)) * (ref2+ref3+ref4+ref5);
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

		if (continu != 0)
		{	
			for (int j = 1; j < n-1; j++) {
				for (int i = 1; i < m-1; i++) {
					if (procRank == nb_procs){
						procRank = 2;
					}
					refs[0] = i;
					refs[1] = j;
					refs[2] = matrix[get_offset(i, j, m, n)];
					refs[3] = matrix[get_offset(i-1, j, m, n)];
					refs[4] = matrix[get_offset(i+1, j, m, n)];
					refs[5] = matrix[get_offset(i, j-1, m, n)];
					refs[6] = matrix[get_offset(i, j+1, m, n)];
					refs[7] = 1;

					MPI_Send(refs, 8, MPI_DOUBLE, procRank, 0, MPI_COMM_WORLD);
					procRank++;
				}
			}	
		}	
	}

	refs[7] = 0;
	for (int p = 2; p < nb_procs; ++p)
	{
		MPI_Send(refs, 8, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);			
	}
}

void processResult(double *matrix, int m, int n, int np) {
	int tailleMatrix = m * n, ri, rj;
	double matrixCurrent[tailleMatrix+1], result[3], res;

	copyMatrix(matrix, matrixCurrent, n, m);
	matrixCurrent[tailleMatrix] = 1;
	
	for (int k = 1; k < np; k++) {
		MPI_Send(matrixCurrent, tailleMatrix+1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
		for (int j = 1; j < n-1; j++) {
			for (int i = 1; i < m-1; i++) {
				MPI_Recv(result, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				ri = (int)(result[0]);
				rj = (int)(result[1]);
				res = result[2];
				matrix[get_offset(ri, rj, m, n)] = res;
				matrixCurrent[get_offset(ri, rj, m, n)] = res;
			}
		}
	}
	matrixCurrent[tailleMatrix] = 0;
	MPI_Send(matrixCurrent, tailleMatrix+1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
}

void executeTask(int m, int n, double td, double h) {
	double refs[8], ref1, ref2, ref3, ref4, ref5, result[3], continu = 1;
	int i, j; 

	while(continu == 1) {
		MPI_Recv(refs, 8, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		continu = refs[7];
		
		if (continu != 0)
		{
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
void init_matrix(double *matrix, int m, int n) {
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < m; i++) {
			matrix[get_offset(i, j, m, n)] = i * (m - i - 1) * j * (n - j - 1);
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
void print_matrix_at_k(double *matrix, int m, int n) {
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < m; i++){
			printf("%9.3f \t", matrix[get_offset(i, j, m, n)]);
		}
		printf("\n");
	}
	printf("\n\n");
}

void copyMatrix(double *matrixSource, double *matrixDest, int n, int m) {
for (int j = 0; j < n; j++) {
		for (int i = 0; i < m; i++) {
			matrixDest[get_offset(i, j, m, n)] = matrixSource[get_offset(i, j, m, n)];
		}
	}
}