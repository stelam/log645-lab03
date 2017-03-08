#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include <mpi.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstddef>
#include <vector>

#define SLEEP_TIME 5

void initMPI(int *argc, char ***argv, int &number_of_processors, int &processor_rank);
void print_matrix_at_k(double *matrix, int m, int n, int np, int k);
void init_matrix(double *matrix, int m, int n, int np);
int get_offset(int k, int i, int j, int m, int n);
void parallel(int number_of_processors, int processor_rank, double *matrix, int m, int n, int np, double td, double h, int nb_procs);
void sequential(double *matrix, int m, int n, int np, double td, double h);
void start_timer(double *time_start);
double stop_timer(double *time_start);


bool is_offset_on_border(int offset, int m, int n);
int* get_i_j_from_offset(int offset, int m, int n);
std::vector<int> get_neighbors_offsets(int offset, int nb_procs, int m, int n);
std::vector<int> get_managed_cells_offsets(int proc_rank, int nb_procs, int k, int m, int n);

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
		time_seq = stop_timer(&time_start);
		printf("Matrice finale : \n");
		print_matrix_at_k(matrix, M, N, NP, NP - 1);
		printf("================================================================================ \n\n\n");

		// parallel
		printf("\n================================== Parallèle ================================== \n");
		init_matrix(matrix, M, N, NP);
		printf("Matrice initiale : \n");
		print_matrix_at_k(matrix, M, N, NP, 0);
		MPI_Barrier(MPI_COMM_WORLD);
		start_timer(&time_start);

	}



	parallel (number_of_processors, processor_rank, matrix, M, N, NP, TD, H, NB_PROCS);


	if (processor_rank == 0) {
		time_parallel = stop_timer(&time_start);
		printf("Matrice finale : \n");
		print_matrix_at_k(matrix, M, N, NP, NP - 1);
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
void parallel(int number_of_processors, int processor_rank, double *matrix, int m, int n, int np, double td, double h, int nb_procs) {

	double ref1, ref2, ref3, ref4, ref5;
	std::vector<int> managed_cells_offsets;
	std::vector<int> neighbors_offsets;
	int offset, i, j, target_proc_rank, received_msg, expected_nb_msg;
	int matrix_size = m * n;
	double neighbors_sum;
	std::vector<double> messages[4];
	std::vector<double> managed_values(((matrix_size + number_of_processors - 1) / number_of_processors) * 2); //[value, offset]
	std::vector<double> message(3); //[value, offset, proc_rank]
	//double message[3];

	for (int k = 1; k < np; k++) {

		managed_cells_offsets = get_managed_cells_offsets(processor_rank, number_of_processors, k, m, n);
		neighbors_offsets = get_neighbors_offsets(offset, number_of_processors, m, n);



		// go through all the managed cells
		for (int c; c < managed_cells_offsets.size(); c++) {
			/*printf("IN %d\n", managed_cells_offsets.size());*/
			offset = managed_cells_offsets[c];
			i = offset % m;
			j = offset / m;
			managed_values[(2*c)+1] = offset;

			printf("IN %d\n", k);

			// if the current cell is in border, just set it to 0
			if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
				managed_values[2*c] = 0;
				printf("IN %d\n", k);
			} else {
				printf("IN %d\n", k);
				// if in first iteration, all values are already available from initialization
				if (k == 1) {
					ref1 = matrix[get_offset(k-1, i, j, m, n)];
					ref2 = matrix[get_offset(k-1, i-1, j, m, n)];
					ref3 = matrix[get_offset(k-1, i+1, j, m, n)];
					ref4 = matrix[get_offset(k-1, i, j-1, m, n)];
					ref5 = matrix[get_offset(k-1, i, j+1, m, n)];

					managed_values[2*c] = ((1 - ((4 * td) / (h*h))) * ref1) + (td / (h*h)) * (ref2+ref3+ref4+ref5);



				
				// for the following iterations ...
				} else {
					// wait for neighbors messages
					received_msg = 0;
					expected_nb_msg = 0;
					neighbors_sum = 0;

					for (int d = 0; d < 4; d++) {
						if (neighbors_offsets[d] >= 0) {
							expected_nb_msg++;
						}
					}

					while (received_msg < expected_nb_msg) {
						MPI_Recv(&message, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						messages[received_msg] = message;
						received_msg++;
					}

					for (int d = 0; d < received_msg; d++) {
						neighbors_sum += messages[d][0];
					}

					// calculate new value
					ref1 = managed_values[2*c]; // previous k
					managed_values[2*c] = ((1 - ((4 * td) / (h*h))) * ref1) + (td / (h*h)) * (neighbors_sum);

				}


				// construct message to send
				message[0] = managed_values[2*c]; // the value to send
				message[1] = offset;
				message[2] = processor_rank;

				// send message to neighbors
				for (int d = 0; d < 4; d++) {
					if (neighbors_offsets[d] >= 0) { // send messages only to neighbors that aren't in borders
						target_proc_rank = neighbors_offsets[d] % number_of_processors;
						MPI_Send(&message, 3, MPI_DOUBLE, target_proc_rank, 0, MPI_COMM_WORLD);
					}

				}

			}

		}	
	}

	// printf("Fini pour proc %6.1f\n", managed_values[0]);
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


/*
* Function: get_managed_cells_offsets
* ----------------------------
*	For a given proc rank, returns the corresponding managed cells, using a cyclic distribution
*
*   proc_rank: the rank of the processor
*
*/
std::vector<int> get_managed_cells_offsets(int proc_rank, int nb_procs, int k, int m, int n) {
	int matrix_size = m * n;
	int nb_managed_cells = 1;
	int i, j;
	std::vector<int> managed_cells;

	i = proc_rank % m;
	j = proc_rank / m;

	// more tasks than cpus
	if (matrix_size > nb_procs) {
		nb_managed_cells = (matrix_size + nb_procs - 1) / nb_procs;

		for (int c; c < nb_managed_cells; c++) {
			managed_cells.push_back(get_offset(k, i, j, m, n) + (c * nb_procs));
		}

	// less tasks than cpus (could be optimized? or merge with one of the
	// other conditions later if no optimization)
	} else if (matrix_size < nb_procs) {
		managed_cells.push_back(get_offset(k, i, j, m, n));

	// equal number of tasks and cpus
	} else {
		managed_cells.push_back(get_offset(k, i, j, m, n));
	}

	return managed_cells;
}

// for each managed cell, we need to find the related neighbors
// the proc_rank can be determined by the offset by doing offset % nb_procs
std::vector<int> get_neighbors_offsets(int offset, int nb_procs, int m, int n) {
	std::vector<int> neighbors(4);

	// identify neighbors
	neighbors[0] = offset - m;
	neighbors[1] = offset - 1;
	neighbors[2] = offset + 1;
	neighbors[3] = offset + m;

	// detect borders
	for (int c = 0; c < 4; c++) {
		// if the neighbor is on a border
		if (is_offset_on_border(neighbors[c], m, n)) {
			neighbors[c] = -1;

		// or if the current cell is on a border
		} else if (is_offset_on_border(offset, m, n)) {
			neighbors[c] = -1;
		}
	}

	return neighbors;

}


int* get_i_j_from_offset(int offset, int m, int n) {
	int i, j;
	i = offset % m;
	j = offset / m;
	int ij[2] = {i, j};
	return ij;
}

bool is_offset_on_border(int offset, int m, int n) {
	int i, j;
	i = offset % m;
	j = offset / m;
	return (i == 0 || i == m -1 || j == 0 || j == n - 1);
}