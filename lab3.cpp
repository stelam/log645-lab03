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
#define MAX_NB_NEIGHBORS 4

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
std::vector<int> get_neighbors_offsets(int offset, int nb_procs, int proc_rank, int m, int n);
std::vector<int> get_managed_cells_offsets(int proc_rank, int nb_procs, int k, int m, int n);
std::vector<int> get_target_neighbors_proc_rank(int offset, int proc_rank, int nb_procs, int m, int n);

int main(int argc, char** argv) {
	const int N = atoi(argv[1]);
	const int M = atoi(argv[2]);
	const int NP = atoi(argv[3]);
	const double TD = atof(argv[4]);
	const double H = atof(argv[5]);
	const int NB_PROCS = atoi(argv[6]);

	double matrix[M * N * NP];
	int processor_rank;
	int number_of_processors = NB_PROCS;
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
		start_timer(&time_start);

	} else {
		init_matrix(matrix, M, N, NP);	
	}


	MPI_Barrier(MPI_COMM_WORLD);
	if (NB_PROCS > (M*N))
		number_of_processors = M * N;
	else
		number_of_processors = NB_PROCS;

	if (processor_rank < (M * N) && processor_rank < NB_PROCS){
		parallel(number_of_processors, processor_rank, matrix, M, N, NP, TD, H, NB_PROCS);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

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
				if (i == 0 || i == m -1 || j == 0 || j == n - 1) {
					matrix[get_offset(k, i, j, m, n)] = 0;
				} else {
					ref1 = matrix[get_offset(k-1, i, j, m, n)];
					ref2 = matrix[get_offset(k-1, i-1, j, m, n)];
					ref3 = matrix[get_offset(k-1, i+1, j, m, n)];
					ref4 = matrix[get_offset(k-1, i, j-1, m, n)];
					ref5 = matrix[get_offset(k-1, i, j+1, m, n)];
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
void parallel(int number_of_processors, int processor_rank, double matrix[], int m, int n, int np, double td, double h, int nb_procs) {

	double ref1, ref2, ref3, ref4, ref5;
	std::vector<int> managed_cells_offsets;
	std::vector<int> neighbors_offsets, target_neighbors_proc_rank;
	int offset, i, j, target_proc_rank, expected_nb_msg, current_k_msg_count;
	int matrix_size = m * n;
	int received_msg = 0;
	double neighbors_sum;
	std::vector<std::vector<double> > messages(4*np);
	std::vector<double> managed_values(((matrix_size + number_of_processors - 1) / number_of_processors)); //[value, offset]
	std::vector<double> message(4); //[value, offset, proc_rank, k]
	//double message[3];

	
	for (int k = 1; k < np; k++) {
		// MPI_Barrier(MPI_COMM_WORLD);

		if (processor_rank < (m * n) || number_of_processors < (m*n)) {

			managed_cells_offsets = get_managed_cells_offsets(processor_rank, number_of_processors, k, m, n);
			


			// printf("WAITING MSG 1 %6.1d\n", k);			

			// printf("WAITING MSG 1 %6.1f\n", matrix[12]);

			

			// go through all the managed cells of current k
			for (int c = 0; c < (int)managed_cells_offsets.size(); c++) {
				/*printf("IN %d\n", managed_cells_offsets.size());*/

				offset = managed_cells_offsets[c];
				i = offset % m;
				j = (offset - ((k)*matrix_size)) / m;
				// if (j > 4)
				// 	printf("%d for k=%d : (%d - (%d * %d)) / % d\n", j,k, offset, k, matrix_size, m);
				// printf("WAITING MSG 1 %6.1d\n", managed_values.size());	

				neighbors_offsets = get_neighbors_offsets(offset, processor_rank, number_of_processors, m, n);
				target_neighbors_proc_rank = get_target_neighbors_proc_rank(offset, processor_rank, number_of_processors, m, n);


				// printf("WAITING MSG 1 %6.1f\n", managed_values[1]);

				// printf("WAITING MSG 1 %6.1f\n", matrix[12]);

				// if the current cell is in border, just set it to 0
				if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
					managed_values[c] = 0;
					// printf("WAITING MSG 2 %6.1f\n", matrix[12]);
				} else {
					//printf("WAITING MSG 3 %6.1d\n", k);
					// if in first iteration, all values are already available from initialization
					if (k == 1) {
						ref1 = matrix[get_offset(k-1, i, j, m, n)];
						ref2 = matrix[get_offset(k-1, i-1, j, m, n)];
						ref3 = matrix[get_offset(k-1, i+1, j, m, n)];
						ref4 = matrix[get_offset(k-1, i, j-1, m, n)];
						ref5 = matrix[get_offset(k-1, i, j+1, m, n)];

						//printf("WAITING MSG 1 %6.1d\n", get_offset(k-1, i, j, m, n));
						

						managed_values[c] = ((1 - ((4 * td) / (h*h))) * ref1) + (td / (h*h)) * (ref2+ref3+ref4+ref5);
						usleep(SLEEP_TIME);
						

					
					// for the following iterations ...
					} else {
						// wait for neighbors messages
						expected_nb_msg = neighbors_offsets[4];
						neighbors_sum = 0;
						bool wait_message = true;
						current_k_msg_count = 0;

						// verify that we do need to receive new messages
						// would be better if we change the index of messages to correspond to their related k
						for (int d = 0; d < expected_nb_msg * k; d++){
							if (messages[d].size() > 0){
								if (messages[d][3] == k - 1) {
									current_k_msg_count++;
									neighbors_sum += messages[d][0];
								}
								if (current_k_msg_count == expected_nb_msg) {
									wait_message = false;
									break;
								}							
							}

						}

						while (wait_message && expected_nb_msg > 0){
							// printf("proc %d Expecting %d messages\n", processor_rank, expected_nb_msg);
							message.resize(4); // not sure if useful
							MPI_Recv(message.data(), 4, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
							// printf("WAITING MSG 1 %6.1f\n", message[2]);
							messages[received_msg] = message;
							
							

							if (messages[received_msg][3] == k - 1) {
								neighbors_sum += messages[received_msg][0];
								current_k_msg_count++;
							}

							if (current_k_msg_count == expected_nb_msg){
								wait_message = false;
							}
							received_msg++;

							printf("I am proc %d and I received %d/%d messages for ijk %d,%d,%d at offset %d \n", processor_rank, current_k_msg_count, expected_nb_msg, i, j,k, offset);
						}




						// calculate new value
						ref1 = managed_values[c]; // previous k
						managed_values[c] = ((1 - ((4 * td) / (h*h))) * ref1) + (td / (h*h)) * (neighbors_sum);
						usleep(SLEEP_TIME);

					}


					// construct message to send
					message[0] = managed_values[c]; // the value to send
					message[1] = offset * 1.0;
					message[2] = processor_rank * 1.0;
					message[3] = k * 1.0;
					// printf("%f\n", message[3]);
					// send message to neighbors

					for (int d = 0; d < MAX_NB_NEIGHBORS; d++) {
						if (target_neighbors_proc_rank[2*d] >= 0) { // send messages only to neighbors that aren't in borders
							target_proc_rank = target_neighbors_proc_rank[2*d+1];
							printf("I am %d at offset %d and I'm SENDING to %d for offset %d at k=%d\n", processor_rank, offset, target_proc_rank, target_neighbors_proc_rank[2*d],k);
							MPI_Send(message.data(), 4, MPI_DOUBLE, target_proc_rank, 0, MPI_COMM_WORLD);
						}

					}
				}

			}	
			// printf("%d\n", processor_rank);
		}

		// printf("WAITING MSG 1 %6.1d\n", k);
	}

	// printf("%d\n", processor_rank);
	if (processor_rank != 0) {
		// printf("%d\n", processor_rank);
		MPI_Send(message.data(), 4, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

	} else {
		received_msg = 0;
		while (received_msg < number_of_processors - 1) {
			// printf("%d < %d?\n", received_msg, number_of_processors);
			MPI_Recv(message.data(), 4, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			matrix[(int)message[1]] = message[0];
			received_msg++;
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
					matrix[get_offset(k, i, j, m, n)] = 0;
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
			printf("%9.6f \t", matrix[get_offset(k, i, j, m, n)]);
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
		int managed_offset = (k*matrix_size) - ((matrix_size - nb_procs) * k) + proc_rank;
		if (managed_offset < (k * matrix_size)) {
			managed_offset += nb_procs;
		}
		while (managed_offset < (k+1) * matrix_size) {
			managed_cells.push_back(managed_offset);
			managed_offset += nb_procs;
		}

	// less tasks than cpus (could be optimized? or merge with one of the
	// other conditions later if no optimization)
	} else if (matrix_size < nb_procs) {
		if (proc_rank <= matrix_size){
			// printf("ij %d,%d for proc %d at managed offset %d\n", i, j, proc_rank, get_offset(k, i, j, m, n));
			managed_cells.push_back(get_offset(k, i, j, m, n));			
		}


	// equal number of tasks and cpus
	} else {
		managed_cells.push_back(get_offset(k, i, j, m, n));
	}

	return managed_cells;
}

// for each managed cell, we need to find the related neighbors
// the neighbors' proc_rank can be determined by the offset by doing offset % nb_procs
std::vector<int> get_neighbors_offsets(int offset, int proc_rank, int nb_procs, int m, int n) {
	std::vector<int> neighbors(5);
	int nb_valid_neighbors = 4;

	int i, j, k;
	k = offset / (m * n);
	i = offset % m;
	j = (offset - ((k)*m*n)) / m;

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
			nb_valid_neighbors--;

		// or if the current cell is on a border
		} else if (is_offset_on_border(offset, m, n)) {
			neighbors[c] = -1;
			nb_valid_neighbors--;
		}
	}
	neighbors[4] = nb_valid_neighbors;
/*	if (proc_rank == 12){
		printf("Valid neighbors: %d for ij %d, %d of proc# %d of offset %d \n", nb_valid_neighbors, i, j, proc_rank, offset);
	}*/
	return neighbors;

}


// for each managed cell, we need to find the related neighbors
// the neighbors' proc_rank can be determined by the offset by doing offset % nb_procs
std::vector<int> get_target_neighbors_proc_rank(int offset, int proc_rank, int nb_procs, int m, int n) {
	std::vector<int> neighbors(9);
	int nb_valid_neighbors = 4;

	int i, j, k, neighbor_delta;
	k = offset / (m * n);
	i = offset % m;
	j = (offset - ((k)*m*n)) / m;

	if (m*n > nb_procs) {
		neighbor_delta = (m*n) - nb_procs;
	} else {
		neighbor_delta = 0;
	}

	// identify neighbors
	neighbors[0] = offset - m;
	neighbors[1] = (neighbors[0] % nb_procs) + neighbor_delta;
	neighbors[2] = offset - 1;
	neighbors[3] = (neighbors[2] % nb_procs) + neighbor_delta;
	neighbors[4] = offset + 1;
	neighbors[5] = (neighbors[4] % nb_procs) + neighbor_delta;
	neighbors[6] = offset + m;
	neighbors[7] = (neighbors[6] % nb_procs) + neighbor_delta;

	// detect borders
	for (int c = 0; c < 4; c++) {
		// if the neighbor is on a border
		if (is_offset_on_border(neighbors[2*c], m, n)) {
			neighbors[2*c] = -1;
			nb_valid_neighbors--;

		// or if the current cell is on a border
		} else if (is_offset_on_border(offset, m, n)) {
			neighbors[2*c] = -1;
			nb_valid_neighbors--;
		}
	}
	neighbors[8] = nb_valid_neighbors;
/*	if (proc_rank == 12){
		printf("Valid neighbors: %d for ij %d, %d of proc# %d of offset %d \n", nb_valid_neighbors, i, j, proc_rank, offset);
	}*/
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
	int i, j, k;

	k = offset / (m * n);
	i = offset % m;
	j = (offset - ((k)*m*n)) / m;
	return (i == 0 || i == m -1 || j == 0 || j == n - 1);
}