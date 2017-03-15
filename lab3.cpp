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
#define MAX_NB_DEPENDENCIES 5

#define TOP_DEPENDENCY 0
#define RIGHT_DEPENDENCY 1
#define BOTTOM_DEPENDENCY 2
#define LEFT_DEPENDENCY 3
#define Z_DEPENDENCY 4


void initMPI(int *argc, char ***argv, int &nb_procs, int &proc_rank);
void print_matrix(double *matrix, int m, int n);
void init_matrix(double *matrix, int m, int n);
int get_offset(int i, int j, int m, int n);
int get_offset_from_inner_i_j(int inner_i, int inner_j, int m, int n);
void parallel(int nb_procs, int proc_rank, double *matrix, int m, int n, int np, double td, double h);
void sequential(double *matrix, int m, int n, int np, double td, double h);
void start_timer(double *time_start);
double stop_timer(double *time_start);
void copyMatrix(double *matrixSource, double *matrixDest, int n, int m);

std::vector<std::vector<std::vector<int> > > get_dependency_procs(int *offset_range, int proc_rank, int nb_procs, int m, int n);
std::vector<int> get_dependency_directions(int local_offset, int m, int n, bool rows);
void get_managed_cells_by_k(int (&offset_range)[2], int proc_rank, int nb_procs, int m, int n);

int main(int argc, char** argv) {
	const int N = atoi(argv[1]);
	const int M = atoi(argv[2]);
	const int NP = atoi(argv[3]);
	const double TD = atof(argv[4]);
	const double H = atof(argv[5]);
	const int NB_PROCS = atoi(argv[6]);

	double matrix[M * N];
	int nb_procs, proc_rank;
	double time_seq, time_parallel, acc, time_start;


	initMPI(&argc, &argv, nb_procs, proc_rank);

	if(proc_rank == 0) {
		// sequential
		printf("\n================================== Séquentiel ================================== \n");
		init_matrix(matrix, M, N);
		printf("Matrice initiale : \n");
		print_matrix(matrix, M, N);
		start_timer(&time_start);
		sequential(matrix, M, N, NP, TD, H);
		time_seq = stop_timer(&time_start);
		printf("Matrice finale : \n");
		print_matrix(matrix, M, N);
		printf("================================================================================ \n\n\n");

		// parallel
		printf("\n================================== Parallèle ================================== \n");
		init_matrix(matrix, M, N);
		printf("Matrice initiale : \n");
		print_matrix(matrix, M, N);

	} else {
		init_matrix(matrix, M, N);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	if (NB_PROCS > (M-2)*(N-2))
		nb_procs = (M-2) * (N-2);
	else
		nb_procs = NB_PROCS;

	if (proc_rank == 0) {
		start_timer(&time_start);
	}

	if (proc_rank < nb_procs) {
		parallel(nb_procs, proc_rank, matrix, M, N, NP, TD, H);
	}
	

	// MPI_Barrier(MPI_COMM_WORLD);

	// printf("%d done\n", proc_rank);

	if (proc_rank == 0) {
		time_parallel = stop_timer(&time_start);
		printf("Matrice finale : \n");
		print_matrix(matrix, M, N);
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
*	nb_procs: pointer to the variable
*	nb_procs: pointer to the variable
*
*/
void initMPI(int *argc, char ***argv, int &nb_procs, int &proc_rank) {
	MPI_Init(&*argc, &*argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
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

int get_offset_from_inner_i_j(int inner_i, int inner_j, int m, int n) {
	return ((inner_j+1) * m) + inner_i + 1;
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


/*
* Function: first_parallel_operation
* ----------------------------
*	Performs the first operation of the lab
*
*   nb_procs: the number of available processors for the parallel processing
*   proc_rank: the rank of the current processor
*	matrix: the matrix that will contain the final results
*	k: the number of alterations to perform
*	starting_value: the initialization value for the initial matrix cells
*
*/
void parallel(int nb_procs, int proc_rank, double matrix[], int m, int n, int np, double td, double h) {

	double top_reference_value, right_reference_value, bottom_reference_value, left_reference_value, z_reference_value;
	// std::vector<int> managed_cells_offsets(2); // inclusive
	int managed_cells_offsets[2] = {-1, -1};

	int local_offset, global_offset, global_i, global_j, inner_i, inner_j, previous_managed_value_inner_i, previous_managed_value_inner_j, nb_received_msg, nb_expected_msg, nb_received_final_msg, counter;
	int inner_matrix_size = (m-2) * (n-2); // inner matrix size
	double dependencies_value_sum, current_cell_value;
	int msg_indexes_to_delete[MAX_NB_DEPENDENCIES] = {-1,-1,-1,-1,-1};

	std::vector<std::vector<std::vector<double> > > messages((np+1));
	// std::vector<double> managed_values(((matrix_size + nb_procs - 1) / nb_procs)); //[value, local_offset, k]
	std::vector<std::vector<std::vector<double> > > managed_values(np); // [k][managed_value_index][value, local_offset]
	std::vector<double> message(5);
	std::vector<double> received_message(5);

	// std::vector<std::vector<int> > dependencies_proc_ranks, target_dependant_proc_ranks;
	std::vector<std::vector<std::vector<int> > >  dependencies_proc_ranks; // [cell_index][dependency_direction][proc_rank, proc_offset]
	std::vector<int> dependency_directions;

	// std::vector<int> self_dependencies_directions(MAX_NB_DEPENDENCIES);
	int self_dependencies_directions[MAX_NB_DEPENDENCIES];

	MPI_Request request, request2;
	MPI_Status status;

	nb_received_final_msg = 0;
	// printf("WAITING MSG 1 %6.1f\n", matrix[12]);

	get_managed_cells_by_k(managed_cells_offsets, proc_rank, nb_procs, m, n);
	dependencies_proc_ranks = get_dependency_procs(managed_cells_offsets, proc_rank, nb_procs, m, n);

	// printf("I a %d, managing %d to %d\n", proc_rank, managed_cells_offsets[0], managed_cells_offsets[1]);

	if (managed_cells_offsets[0] > -1){
		for (int k = 1; k < np; k++) {
			// MPI_Barrier(MPI_COMM_WORLD);
			
			// go through all the managed cells of current k
			for (int cell_index = 0; cell_index <= managed_cells_offsets[1] - managed_cells_offsets[0]; cell_index++) {
				/*printf("IN %d\n", managed_cells_offsets.size());*/

				local_offset = managed_cells_offsets[0] + cell_index;
				inner_i = local_offset % (m-2);
				inner_j = local_offset / (m-2);


				// wait for neighbors messages
				nb_received_msg = 0;
				dependencies_value_sum = 0;
				nb_expected_msg = 0;
				top_reference_value = 0;
				right_reference_value = 0;
				bottom_reference_value = 0;
				left_reference_value = 0;
				z_reference_value = 0;

				// self_dependencies_directions.resize(5);
				// calculate the number of expected messages coming from dependencies

				std::fill_n(self_dependencies_directions, 5,-1);
				for (int d = 0; d < MAX_NB_DEPENDENCIES; d++) {
					if (dependencies_proc_ranks[cell_index][d][0] != proc_rank && dependencies_proc_ranks[cell_index][d][0] > -1){
						nb_expected_msg++;
						// if (proc_rank == 1)
						// 	printf("I'm offset %d I have a dependency %d\n", local_offset, d);

					}
					else
						self_dependencies_directions[d] = 1;
				}

				
				//printf("WAITING MSG 3 %6.1d\n", k);
				// if in first iteration, all values are already available from initialization
				if (k == 1) {
					dependency_directions = get_dependency_directions(local_offset, m, n, (m<=n));
					// printf("dd %d\n", dependency_directions[TOP_DEPENDENCY]);
					z_reference_value = matrix[get_offset_from_inner_i_j(inner_i, inner_j, m, n)];
					if (dependency_directions[TOP_DEPENDENCY] == 1) top_reference_value = matrix[get_offset_from_inner_i_j(inner_i, inner_j+1, m, n)];
					if (dependency_directions[RIGHT_DEPENDENCY] == 1) right_reference_value = matrix[get_offset_from_inner_i_j(inner_i+1, inner_j, m, n)];
					if (dependency_directions[BOTTOM_DEPENDENCY] == 1) bottom_reference_value = matrix[get_offset_from_inner_i_j(inner_i, inner_j-1, m, n)];
					if (dependency_directions[LEFT_DEPENDENCY] == 1) left_reference_value = matrix[get_offset_from_inner_i_j(inner_i-1, inner_j, m, n)];
					//printf("WAITING MSG 1 %6.1d\n", get_offset(k-1, i, j, m, n));
					// printf("dd %d\n", dependency_directions[LEFT_DEPENDENCY]);
					current_cell_value = ((1 - ((4 * td) / (h*h))) * z_reference_value) + (td / (h*h)) * (top_reference_value+right_reference_value+bottom_reference_value+left_reference_value);
					
					// if (proc_rank == 0)
					// printf("I am proc %d at k = %d, offset = %d, my dependencies : %d %d %d %d\n", proc_rank, k, local_offset, dependency_directions[TOP_DEPENDENCY], dependency_directions[RIGHT_DEPENDENCY], dependency_directions[BOTTOM_DEPENDENCY], dependency_directions[LEFT_DEPENDENCY]);
					// // store managed values
					managed_values[k].push_back({current_cell_value, local_offset * 1.0});
					// if (proc_rank == 2)
					// printf("I am proc %d, inneri %d innerj %d, inner offset %d, outer offset %d\n", proc_rank, inner_i, inner_j, local_offset, get_offset_from_inner_i_j(k-1, inner_i, inner_j, m, n));

					usleep(SLEEP_TIME);
				
				// for the following iterations ...
				} else {

					// printf("%d\n", (int)self_dependencies_directions.size());
					// get the needed values associated with self-dependencies


					// printf("I am proc %d at k = %d, offset = %d, my dependencies : %d %d %d %d\n", proc_rank, k, local_offset, dependency_directions[TOP_DEPENDENCY], dependency_directions[RIGHT_DEPENDENCY], dependency_directions[BOTTOM_DEPENDENCY], dependency_directions[LEFT_DEPENDENCY]);
					for (int d = 0; d < MAX_NB_DEPENDENCIES; d++) {
						for (int managed_value_index = 0; managed_value_index < (int)managed_values[k-1].size(); managed_value_index++) {

							previous_managed_value_inner_i = (int)managed_values[k-1][managed_value_index][1] % (m-2);
							previous_managed_value_inner_j = (int)managed_values[k-1][managed_value_index][1] / (m-2);

							if (self_dependencies_directions[d] == 1 && d == Z_DEPENDENCY){
								if (previous_managed_value_inner_i == inner_i && previous_managed_value_inner_j == inner_j){
									z_reference_value = managed_values[k-1][managed_value_index][0];
								}
							} else if (self_dependencies_directions[d] == 1 && d == TOP_DEPENDENCY){
								if (previous_managed_value_inner_i == inner_i && previous_managed_value_inner_j == inner_j+1){
									top_reference_value = managed_values[k-1][managed_value_index][0];
									dependencies_value_sum += top_reference_value;
								}

							} else if (self_dependencies_directions[d] == 1 && d == RIGHT_DEPENDENCY){
								if (previous_managed_value_inner_i == inner_i+1 && previous_managed_value_inner_j == inner_j){
									right_reference_value = managed_values[k-1][managed_value_index][0];
									dependencies_value_sum += right_reference_value;
								}

							} else if (self_dependencies_directions[d] == 1 && d == BOTTOM_DEPENDENCY){
								if (previous_managed_value_inner_i == inner_i && previous_managed_value_inner_j == inner_j-1){
									bottom_reference_value = managed_values[k-1][managed_value_index][0];
									dependencies_value_sum += bottom_reference_value;
									// printf("hello\n");
								} else {
									// printf("rank %d offset %d k = %d, previ%d prevj %i, i %d j %d\n", proc_rank, local_offset, k, previous_managed_value_inner_i, previous_managed_value_inner_j, inner_i, inner_j);
								}

							} else if (self_dependencies_directions[d] == 1 && d == LEFT_DEPENDENCY){
								if (previous_managed_value_inner_i == inner_i-1 && previous_managed_value_inner_j == inner_j){
									left_reference_value = managed_values[k-1][managed_value_index][0];
									dependencies_value_sum += left_reference_value;
								}
							} 
						}
					}


					// check if we previously received some message(s) for the current k
					// if so, update the quantity of messages (if any) are still expected
					if (messages[k].size() > 0) {
						for (int msg_index = 0; msg_index < (int) messages[k].size(); msg_index++) {
							if ((int)messages[k][msg_index][2] == local_offset) {
								nb_received_msg++;
							}
						}	
					}

					// wait for messages
					// if (proc_rank == 0) {
					// 	printf("receiving for k = %d\n", k);
					// }
					while (nb_received_msg < nb_expected_msg) {
						// printf("Expecting %d messages\n", expected_nb_msg);
						received_message.resize(5);
						MPI_Recv(&received_message.front(), 5, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						// MPI_Irecv(received_message.data(), 5, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
						// store the message

						if (proc_rank == 0) {
							// receiving the final messages while proc is still not done doing its job
							if ((int)received_message[2] == -2) {
								// printf("received final message\n");
								matrix[(int)received_message[1]] = received_message[0];
								nb_received_final_msg++;
							} else {
								messages[(int)received_message[3]].push_back(received_message);
							}
						} else {
							messages[(int)received_message[3]].push_back(received_message);
						}


						if ((int) received_message[3] == k && (int)received_message[2] == local_offset) {
							nb_received_msg++;
							// if (proc_rank == 0)
							// printf("I am %d at k=%d managing offset %d, I received %d/%d\n", proc_rank, k, local_offset, nb_received_msg, nb_expected_msg);
						}

						// if (proc_rank == 0)
						// printf("I am %d at k=%d, I received a message from %d for local_offset = %d at k = %d, but I want one for local_offset = %d \n", proc_rank, k, (int)message[1], (int)message[3], (int)message[2], local_offset);
				
						// if (proc_rank == 0 && nb_received_msg == nb_expected_msg)
						// 	printf("I am %d, I received %d/%d\n", proc_rank, nb_received_msg, nb_expected_msg);

						// if (proc_rank == 0) {
						// 	printf("\treceived a message: from proc_rank %f, for target offset %f, at k = %f \n", message[1], message[2], message[3]);
						// }

						if (proc_rank == 1) {
							// printf("received %d/%d for offset %d at k=%d from proc %f value = %f, target_offset = %f \n", nb_received_msg, nb_expected_msg, local_offset, k, received_message[1], received_message[0], received_message[2]);
							// printf("%f\n", received_message[1]);
							// printf("my last msg at index %d  \t\t\t\t %f, %f, %f: \n", (int)messages[(int)message[3]].size()-1, messages[(int)message[3]][messages[(int)message[3]].size()-1][1], messages[(int)message[3]][messages[(int)message[3]].size()-1][0], messages[(int)message[3]][messages[(int)message[3]].size()-1][2]);
						}

					}

					// calculate the sum of the 4 neighbors
					// and get the z-dependency value (if applicable)
					// counter = 0;
					// std::fill_n(msg_indexes_to_delete, 5,-1);
					if (nb_expected_msg > 0) {
						for (int msg_index = 0; msg_index < (int)messages[k].size(); msg_index++) {
							if ((int)messages[k][msg_index][2] == local_offset) {
								if ((int)messages[k][msg_index][4] != local_offset) {
									dependencies_value_sum += messages[k][msg_index][0];
									// msg_indexes_to_delete[counter] = msg_index;
								} else if ((int)messages[k][msg_index][4] == local_offset) {
									z_reference_value = messages[k][msg_index][0];
									// msg_indexes_to_delete[counter] = msg_index;
								}
								
							}
							// counter++;
						}
					}

					// // delete messages
					// for (int index = 0; index < MAX_NB_DEPENDENCIES; index++) {
					// 	if (msg_indexes_to_delete[index] > -1) {
					// 		// messages[k].erase(messages[k].begin() + msg_indexes_to_delete[index]);
					// 	}
					// }


					// calculate the value for current cell
					current_cell_value = ((1 - ((4 * td) / (h*h))) * z_reference_value) + ((td / (h*h)) * (dependencies_value_sum));
					managed_values[k].push_back({current_cell_value, local_offset * 1.0});
					usleep(SLEEP_TIME);
				}


				// construct message to send
				if (k+1 < np) {
					message.resize(5);
					message[0] = current_cell_value; // the value to send
					message[1] = proc_rank * 1.0;
					message[2] = -1; // target_local_offset, defined later;
					message[3] = k+1; // target_k
					message[4] = local_offset * 1.0; // current local offset

					// if (k>2)
					// 	MPI_Wait(&request, &status);
					// send message to neighbors
					for (int d = 0; d < MAX_NB_DEPENDENCIES; d++) {
						if (dependencies_proc_ranks[cell_index][d][0] >= 0 && dependencies_proc_ranks[cell_index][d][0] != proc_rank) {
							message[2] = dependencies_proc_ranks[cell_index][d][1] * 1.0; // target_local_offset
							// if (proc_rank == 0)		
							// printf("\t\tI am %d at k = %d, managing offset %d, sending to proc %d at offset %d\n", proc_rank, k, local_offset, dependencies_proc_ranks[cell_index][d][0], (int)message[2]);
							MPI_Send(&message.front(), 5, MPI_DOUBLE, dependencies_proc_ranks[cell_index][d][0], 0, MPI_COMM_WORLD);
							// MPI_Isend(message.data(), 5, MPI_DOUBLE, target_dependant_proc_ranks[d][0], 0, MPI_COMM_WORLD, &request);
							// MPI_Wait(&request, &status);

							// MPI_Wait(&request, MPI_STATUS_IGNORE);
						}
					}
				}

			}	
		}

	
		double final_message[5];

		for (int d = 0; d < (int)managed_values[np-1].size(); d++) {
			global_i = (int)managed_values[np-1][d][1] % (m-2);
			global_j = (int)managed_values[np-1][d][1] / (m-2);

			global_offset = get_offset_from_inner_i_j(global_i, global_j, m, n);
			final_message[0] = managed_values[np-1][d][0];
			final_message[1] = global_offset * 1.0;
			final_message[2] = -2 * 1.0;
			final_message[3] = -2 * 1.0;
			final_message[4] = -2 * 1.0;

			if (proc_rank != 0){
				//printf("I am %d sending %f for offset %f\n", proc_rank, final_message[0], final_message[1]);
				MPI_Send(final_message, 5, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			} else {
				matrix[global_offset] = final_message[0];
			}
		}		

		if (proc_rank == 0) {
			nb_received_msg = 0;
			nb_expected_msg = inner_matrix_size - (int)managed_values[np-1].size() - nb_received_final_msg;
			

			while (nb_received_msg < nb_expected_msg) {
				MPI_Recv(final_message, 5, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (final_message[2] == -2){
					matrix[(int)final_message[1]] = final_message[0];
					nb_received_msg++;
				} else {
					printf("received a late message: from proc_rank %f, for target offset %f, at k = %f \n", final_message[1], final_message[2], final_message[3]);
				}

				// printf("received %d/%d\n", nb_received_msg, nb_expected_msg);
			}
		}

	}

	// printf("Fini pour proc %6.1f\n", managed_values[0]);

	// printf("I am %d and i finished\n", proc_rank);

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
* Function: print_matrix
* ----------------------------
*	Prints the last matrix that is the result of a series of operations
*
*   matrix: the 1D array containing the matrix
*
*/
void print_matrix(double *matrix, int m, int n) {
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


void get_managed_cells_by_k(int (&offset_range)[2], int proc_rank, int nb_procs, int m, int n) {
	// int first_proc_rank = (((m-2)*(n-2))*k) % nb_procs;
	int offset_index;
	int first_offset;
	int nb_effective_procs;

	// if (matrix_size > nb_procs) {



		if (m <= n) {
			int nb_rows = ((n-2) + nb_procs - 1) / nb_procs;
			if ((n-2) < nb_procs and proc_rank >= (n-2) || proc_rank * nb_rows >= (n-2)) {
				offset_range[0] = -1;
				offset_range[1] = -1;
			} else {
				int last_nb_rows = nb_rows;
				if (nb_procs * nb_rows > (n-2)) {
					last_nb_rows = (nb_procs * nb_rows) % (n-2);
				}
				first_offset = proc_rank * nb_rows * (m-2);
				offset_range[0] = first_offset;
				if (proc_rank * nb_rows > (n-2)) {
					offset_range[1] = first_offset + (last_nb_rows * (m-2)) - 1;
				} else {
					offset_range[1] = first_offset + (nb_rows * (m-2)) - 1;
				}				
			}

		} else {
			if ((m-2) < nb_procs and proc_rank >= (m-2)) {
				offset_range[0] = -1;
				offset_range[1] = -1;
			} else {
				int nb_cols = ((m-2) + nb_procs - 1) / nb_procs;
				int last_nb_cols = nb_cols;
				if (nb_procs * nb_cols > (m-2)) {
					last_nb_cols = (nb_procs * nb_cols) % (m-2);
				}
				first_offset = proc_rank * nb_cols * (n-2);
				offset_range[0] = first_offset;
				if (proc_rank * nb_cols > (m-2)) {
					offset_range[1] = first_offset + (last_nb_cols * (n-2)) - 1;
				} else {
					offset_range[1] = first_offset + (nb_cols * (n-2)) - 1;
				}				
			}

		}

	/*} else {
		// printf("ij %d,%d for proc %d at managed offset %d\n", i, j, proc_rank, get_offset(k, i, j, m, n));
		managed_cells.push_back(proc_rank);			

	}*/

}


std::vector<std::vector<std::vector<int> > > get_dependency_procs(int *offset_range, int proc_rank, int nb_procs, int m, int n) {
	std::vector<std::vector<int> > dependencies = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}; // [proc_rank, proc_local_offset]
	std::vector<int> dependency_directions;
	std::vector<std::vector<std::vector<int> > > all_dependencies;

	// if (proc_rank == 0) {
	// 	printf("from %d to %d\n", offset_range[0], offset_range[1]);
	// }

	int counter = 0;
	if (offset_range[0] > -1) {
		if (m <= n) {
			for (int offset_index = offset_range[0]; offset_index <= offset_range[1]; offset_index++) {
				dependency_directions = get_dependency_directions(offset_index, m, n, true);
				dependencies = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}};

				dependencies[Z_DEPENDENCY][0] = proc_rank;
				dependencies[Z_DEPENDENCY][1] = offset_index;

				if (dependency_directions[LEFT_DEPENDENCY] == 1) {
					dependencies[LEFT_DEPENDENCY][0] = proc_rank;
					dependencies[LEFT_DEPENDENCY][1] = offset_index - 1;
				}

				if (dependency_directions[TOP_DEPENDENCY] == 1) {
					// 17 <= (23 - 6)
					if (offset_index <= (offset_range[1] - (m-2))) {
						dependencies[TOP_DEPENDENCY][0] = proc_rank;
					} else {
						dependencies[TOP_DEPENDENCY][0] = proc_rank+1;
					}
					dependencies[TOP_DEPENDENCY][1] = offset_index + (m-2);
				}

				if (dependency_directions[RIGHT_DEPENDENCY] == 1) {
					dependencies[RIGHT_DEPENDENCY][0] = proc_rank;
					dependencies[RIGHT_DEPENDENCY][1] = offset_index + 1;
				}

				if (dependency_directions[BOTTOM_DEPENDENCY] == 1) {
					if (offset_index < (offset_range[0] + (m-2))){
						dependencies[BOTTOM_DEPENDENCY][0] = proc_rank - 1;
					} else {
						dependencies[BOTTOM_DEPENDENCY][0] = proc_rank;
					}
					dependencies[BOTTOM_DEPENDENCY][1] = offset_index - (m-2);
				}

				all_dependencies.push_back(dependencies);
				counter++;
			}
		} else {
			for (int offset_index = offset_range[0]; offset_index <= offset_range[1]; offset_index++) {
				dependency_directions = get_dependency_directions(offset_index, m, n, true);
				dependencies = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}};

				dependencies[Z_DEPENDENCY][0] = proc_rank;
				dependencies[Z_DEPENDENCY][1] = offset_index;

				if (dependency_directions[LEFT_DEPENDENCY] == 1) {
					if (offset_index <= (offset_range[0] + (n-2))) {
						dependencies[LEFT_DEPENDENCY][0] = proc_rank - 1;
					} else {
						dependencies[LEFT_DEPENDENCY][0] = proc_rank;
					}
					dependencies[LEFT_DEPENDENCY][1] = offset_index - (n-2);
				}

				if (dependency_directions[TOP_DEPENDENCY] == 1) {
					dependencies[TOP_DEPENDENCY][0] = proc_rank;
					dependencies[TOP_DEPENDENCY][1] = offset_index + 1;
				}

				if (dependency_directions[RIGHT_DEPENDENCY] == 1) {
					if (offset_index <= (offset_range[1] - (n-2))) {
						dependencies[RIGHT_DEPENDENCY][0] = proc_rank;
					} else{
						dependencies[RIGHT_DEPENDENCY][0] = proc_rank + 1;
					}
					dependencies[RIGHT_DEPENDENCY][1] = offset_index + (n-2);
				}

				if (dependency_directions[BOTTOM_DEPENDENCY] == 1) {
					dependencies[BOTTOM_DEPENDENCY][0] = proc_rank;
					dependencies[BOTTOM_DEPENDENCY][1] = offset_index - 1;
				}

				all_dependencies.push_back(dependencies);
				counter++;
			}
		}

	}

	return all_dependencies;
}





std::vector<int> get_dependency_directions(int local_offset, int m, int n, bool rows) {
	int i, j;

	if (rows == true) {
		i = local_offset % (m-2);
		j = local_offset / (m-2);		
	} else {
		i = local_offset / (m-2);
		j = local_offset % (m-2);
	}


	std::vector<int> directions = {1, 1, 1, 1, 1};

	if (i == 0) directions[LEFT_DEPENDENCY] = 0;
	if (i == (m-2)-1) directions[RIGHT_DEPENDENCY] = 0;
	if (j == 0) directions[BOTTOM_DEPENDENCY] = 0;
	if (j == (n-2)-1) directions[TOP_DEPENDENCY] = 0;

	return directions;
}