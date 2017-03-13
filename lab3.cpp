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
void print_matrix_at_k(double *matrix, int m, int n, int np, int k);
void init_matrix(double *matrix, int m, int n, int np);
int get_offset(int k, int i, int j, int m, int n);
void parallel(int nb_procs, int proc_rank, double *matrix, int m, int n, int np, double td, double h);
void sequential(double *matrix, int m, int n, int np, double td, double h);
void start_timer(double *time_start);
double stop_timer(double *time_start);

std::vector<std::vector<int> > get_dependency_procs(int local_offset, int nb_procs, int k, int m, int n, bool backward);
std::vector<bool> get_dependency_directions(int local_offset, int m, int n);
std::vector<int> get_managed_cells_by_k(int proc_rank, int nb_procs, int k, int m, int n);

int main(int argc, char** argv) {
	const int N = atoi(argv[1]);
	const int M = atoi(argv[2]);
	const int NP = atoi(argv[3]);
	const double TD = atof(argv[4]);
	const double H = atof(argv[5]);
	const int NB_PROCS = atoi(argv[6]);

	double matrix[M * N * NP];
	int nb_procs, proc_rank;
	double time_seq, time_parallel, acc, time_start;


	initMPI(&argc, &argv, nb_procs, proc_rank);

	if (proc_rank != 0){
		init_matrix(matrix, M, N, NP);	
	}

	if(proc_rank == 0) {
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

	} 

	// MPI_Barrier(MPI_COMM_WORLD);
	if (NB_PROCS > (M-2)*(N-2))
		nb_procs = (M-2) * (N-2);
	else
		nb_procs = NB_PROCS;

	if (proc_rank < nb_procs && proc_rank < (M-2) * (N-2)) {
		parallel(nb_procs, proc_rank, matrix, M, N, NP, TD, H);
	}
	

	// MPI_Barrier(MPI_COMM_WORLD);

	if (proc_rank == 0) {
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
int get_offset(int k, int i, int j, int m, int n) { 
	return (k * m * n) + (j * m) + i; 
}

int get_offset_from_inner_i_j(int k, int inner_i, int inner_j, int m, int n) {
	return (k * m * n) + ((inner_j+1) * m) + inner_i;
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
*   nb_procs: the number of available processors for the parallel processing
*   proc_rank: the rank of the current processor
*	matrix: the matrix that will contain the final results
*	k: the number of alterations to perform
*	starting_value: the initialization value for the initial matrix cells
*
*/
void parallel(int nb_procs, int proc_rank, double matrix[], int m, int n, int np, double td, double h) {

	double top_reference_value, right_reference_value, bottom_reference_value, left_reference_value, z_reference_value;
	std::vector<int> managed_cells_offsets; // for a given k

	int local_offset, global_offset, global_i, global_j, inner_i, inner_j, current_managed_value_inner_i, current_managed_value_inner_j, nb_received_msg, nb_expected_msg, nb_received_final_msg;
	int inner_matrix_size = (m-2) * (n-2); // inner matrix size
	double dependencies_value_sum, current_cell_value;

	std::vector<std::vector<std::vector<double> > > messages((inner_matrix_size / nb_procs) * (np+1));
	// std::vector<double> managed_values(((matrix_size + nb_procs - 1) / nb_procs)); //[value, local_offset, k]
	std::vector<std::vector<std::vector<double> > > managed_values(np); // [k][managed_value_index][value, local_offset]
	std::vector<double> message(5); //[value, offset, proc_rank]

	std::vector<std::vector<int> > dependencies_proc_ranks, target_dependant_proc_ranks;
	std::vector<bool> dependency_directions;

	// std::vector<int> self_dependencies_directions(MAX_NB_DEPENDENCIES);
	int self_dependencies_directions[MAX_NB_DEPENDENCIES];

	MPI_Request request, request2;
	MPI_Status status;

	// printf("WAITING MSG 1 %6.1f\n", matrix[12]);
	for (int k = 1; k < np; k++) {
		// MPI_Barrier(MPI_COMM_WORLD);
		managed_cells_offsets = get_managed_cells_by_k(proc_rank, nb_procs, k, m, n);
		
		// go through all the managed cells of current k
		for (int cell_index = 0; cell_index < (int)managed_cells_offsets.size(); cell_index++) {
			/*printf("IN %d\n", managed_cells_offsets.size());*/

			local_offset = managed_cells_offsets[cell_index];
			inner_i = local_offset % m;
			inner_j = local_offset / m;

			// if (proc_rank == 3)
			// 	printf("I am proc %d during k=%d, managing offset %d \n", proc_rank, k, local_offset);

			//neighbors_offsets = get_neighbors_offsets(offset, proc_rank, nb_procs, m, n);
	

			// wait for neighbors messages
			nb_received_msg = 0;
			dependencies_value_sum = 0;
			nb_expected_msg = 0;
			top_reference_value = 0;
			right_reference_value = 0;
			bottom_reference_value = 0;
			left_reference_value = 0;
			z_reference_value = 0;

			dependencies_proc_ranks = get_dependency_procs(local_offset, nb_procs, k, m, n, true);
			target_dependant_proc_ranks = get_dependency_procs(local_offset, nb_procs, k, m, n, false);

			// self_dependencies_directions.resize(5);
			// calculate the number of expected messages coming from dependencies

			std::fill_n(self_dependencies_directions, 5,-1);
			for (int d = 0; d < MAX_NB_DEPENDENCIES; d++) {
				// if (proc_rank == 0){
				// 	if (local_offset == 6) {
				// 		printf("\t\t%d\n", dependencies_proc_ranks[d][0]);
				// 	}
				// }

				if (dependencies_proc_ranks[d][0] != proc_rank && dependencies_proc_ranks[d][0] > -1)
					nb_expected_msg++;
				else
					self_dependencies_directions[d] = 1;
			}


			//printf("WAITING MSG 3 %6.1d\n", k);
			// if in first iteration, all values are already available from initialization
			if (k == 1) {
				dependency_directions = get_dependency_directions(local_offset, m, n);
				z_reference_value = matrix[get_offset_from_inner_i_j(k-1, inner_i, inner_j, m, n)];
				if (dependency_directions[TOP_DEPENDENCY] == true) top_reference_value = matrix[get_offset_from_inner_i_j(k-1, inner_i, inner_j+1, m, n)];
				if (dependency_directions[RIGHT_DEPENDENCY] == true) right_reference_value = matrix[get_offset_from_inner_i_j(k-1, inner_i+1, inner_j, m, n)];
				if (dependency_directions[BOTTOM_DEPENDENCY] == true) bottom_reference_value = matrix[get_offset_from_inner_i_j(k-1, inner_i, inner_j-1, m, n)];
				if (dependency_directions[LEFT_DEPENDENCY] == true) left_reference_value = matrix[get_offset_from_inner_i_j(k-1, inner_i-1, inner_j, m, n)];
				//printf("WAITING MSG 1 %6.1d\n", get_offset(k-1, i, j, m, n));

				current_cell_value = ((1 - ((4 * td) / (h*h))) * z_reference_value) + (td / (h*h)) * (top_reference_value+right_reference_value+bottom_reference_value+left_reference_value);
				
				// store managed values
				managed_values[k].push_back({current_cell_value, local_offset * 1.0});

				usleep(SLEEP_TIME);
		
			// for the following iterations ...
			} else {

				if (proc_rank == 0) {
					// printf("%d yay\n", k);

					// nb_received_msg = 0;
					// nb_expected_msg = inner_matrix_size - (int)managed_values[np-1].size();
				}

				// printf("%d\n", (int)self_dependencies_directions.size());
				// get the needed values associated with self-dependencies
				for (int d = 0; d < MAX_NB_DEPENDENCIES; d++) {
					for (int managed_value_index = 0; managed_value_index < (int)managed_values[k-1].size(); managed_value_index++) {

						current_managed_value_inner_i = (int)managed_values[k-1][managed_value_index][1] % (m-2);
						current_managed_value_inner_j = (int)managed_values[k-1][managed_value_index][1] / (m-2);

						if (self_dependencies_directions[d] == 1 && d == Z_DEPENDENCY){
							if (current_managed_value_inner_i == inner_i && current_managed_value_inner_j == inner_j)
								z_reference_value = managed_values[k-1][managed_value_index][0];

						} else if (self_dependencies_directions[d] == 1 && d == TOP_DEPENDENCY){
							if (current_managed_value_inner_i == inner_i && current_managed_value_inner_j == inner_j+1){
								top_reference_value = managed_values[k-1][managed_value_index][0];
								dependencies_value_sum += top_reference_value;
							}

						} else if (self_dependencies_directions[d] == 1 && d == RIGHT_DEPENDENCY){
							if (current_managed_value_inner_i == inner_i+1 && current_managed_value_inner_j == inner_j){
								right_reference_value = managed_values[k-1][managed_value_index][0];
								dependencies_value_sum += right_reference_value;
							}

						} else if (self_dependencies_directions[d] == 1 && d == BOTTOM_DEPENDENCY){
							if (current_managed_value_inner_i == inner_i && current_managed_value_inner_j == inner_j-1){
								bottom_reference_value = managed_values[k-1][managed_value_index][0];
								dependencies_value_sum += bottom_reference_value;
							}

						} else if (self_dependencies_directions[d] == 1 && d == LEFT_DEPENDENCY){
							if (current_managed_value_inner_i == inner_i-1 && current_managed_value_inner_j == inner_j){
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
				while (nb_received_msg < nb_expected_msg) {
					// printf("Expecting %d messages\n", expected_nb_msg);
					message.resize(5);
					MPI_Recv(message.data(), 5, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// MPI_Irecv(message.data(), 5, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
					// store the message

					if (proc_rank == 0) {
						// receiving the final messages while proc is still not done doing its job
						if ((int)message[2] == -1) {
							// printf("received final message\n");
							matrix[(int)message[1]] = message[0];
							nb_received_final_msg++;
						} else {
							messages[(int)message[3]].push_back(message);
						}
					} else {
						messages[(int)message[3]].push_back(message);
					}


					if ((int) message[3] == k && (int)message[2] == local_offset) {
						nb_received_msg++;
						// if (proc_rank == 0)
						// printf("I am %d at k=%d managing offset %d, I received %d/%d\n", proc_rank, k, local_offset, nb_received_msg, nb_expected_msg);
					}

					// if (proc_rank == 0)
					// printf("I am %d at k=%d, I received a message from %d for local_offset = %d at k = %d, but I want one for local_offset = %d \n", proc_rank, k, (int)message[1], (int)message[3], (int)message[2], local_offset);
			

					// printf("I am %d, I received %d/%d\n", proc_rank, nb_received_msg, nb_expected_msg);

				}

				// calculate the sum of the 4 neighbors
				// and get the z-dependency value (if applicable)
				if (nb_expected_msg > 0) {
					for (int msg_index = 0; msg_index < nb_expected_msg; msg_index++) {
						if ((int)messages[k][msg_index][2] == local_offset && (int)messages[k][msg_index][4] != local_offset) {
							dependencies_value_sum += messages[k][msg_index][0];
							// messages[k].erase(messages[k].begin());

						} else if ((int)messages[k][msg_index][4] == local_offset) {
							z_reference_value = messages[k][msg_index][0];
							// messages[k].erase(messages[k].begin());
						}
					}
				}

				// calculate the value for current cell
				current_cell_value = ((1 - ((4 * td) / (h*h))) * z_reference_value) + (td / (h*h)) * (dependencies_value_sum);
				managed_values[k].push_back({current_cell_value, local_offset * 1.0});
				usleep(SLEEP_TIME);

			}


			// construct message to send
			message[0] = current_cell_value; // the value to send
			message[1] = proc_rank * 1.0;
			message[2] = -1; // target_local_offset, defined later;
			message[3] = k+1; // target_k
			message[4] = local_offset * 1.0;

			// if (k>2)
			// 	MPI_Wait(&request, &status);
			// send message to neighbors
			for (int d = 0; d < MAX_NB_DEPENDENCIES; d++) {
				if (target_dependant_proc_ranks[d][0] >= 0 && target_dependant_proc_ranks[d][0] != proc_rank) {
					message[2] = target_dependant_proc_ranks[d][1]; // target_local_offset
					// if (proc_rank == 1)
					// 	printf("\t\tI am %d at k = %d, managing offset %d, sending to proc %d at offset %d\n", proc_rank, k, local_offset, target_dependant_proc_ranks[d][0], (int)message[2]);
					MPI_Send(message.data(), 5, MPI_DOUBLE, target_dependant_proc_ranks[d][0], 0, MPI_COMM_WORLD);
					// MPI_Isend(message.data(), 5, MPI_DOUBLE, target_dependant_proc_ranks[d][0], 0, MPI_COMM_WORLD, &request);
					// MPI_Wait(&request, &status);
				}
			}

		}	
	}

	double final_message[5];

	for (int d = 0; d < (int)managed_values[np-1].size(); d++) {
		global_i = (int)managed_values[np-1][d][1] % (m-2);
		global_j = (int)managed_values[np-1][d][1] / (m-2);

		global_offset = get_offset_from_inner_i_j(np-1, global_i, global_j, m, n);
		final_message[0] = managed_values[np-1][d][0];
		final_message[1] = global_offset * 1.0;
		final_message[2] = -1;
		final_message[3] = -1;

		if (proc_rank != 0){
			// printf("I am %d sending %f for offset %f\n", proc_rank, final_message[0], final_message[1]);
			MPI_Send(final_message, 5, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		} else {
			matrix[global_offset] = final_message[0];
		}
	}


	if (proc_rank == 0) {
		nb_received_msg = 0;
		nb_expected_msg = inner_matrix_size - (int)managed_values[np-1].size() - nb_received_final_msg;
		printf("expecting %d\n", nb_expected_msg);

		while (nb_received_msg < nb_expected_msg) {
			MPI_Recv(final_message, 5, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			matrix[(int)final_message[1]] = final_message[0];
			nb_received_msg++;
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
			printf("%5.1f \t", matrix[get_offset(k, i, j, m, n)]);
		}
		printf("\n");
	}
	printf("\n\n");
}


std::vector<int> get_managed_cells_by_k(int proc_rank, int nb_procs, int k, int m, int n) {
	int first_proc_rank = (((m-2)*(n-2))*k) % nb_procs;
	int offset_index;
	std::vector<int> managed_cells;

	if (proc_rank == first_proc_rank) {
		managed_cells.push_back(0);
	} else {
		if (proc_rank > first_proc_rank)
			managed_cells.push_back(proc_rank - first_proc_rank);
		else
			managed_cells.push_back(nb_procs - abs(first_proc_rank - proc_rank));
	}

	offset_index = managed_cells[0];

	while (offset_index < (m-2)*(n-2)) {
		offset_index += nb_procs;
		if (offset_index < (m-2)*(n-2)){
			managed_cells.push_back(offset_index);
		}
	}

	return managed_cells;
}


std::vector<std::vector<int> > get_dependency_procs(int local_offset, int nb_procs, int k, int m, int n, bool backward) {
	int first_proc_rank;
	std::vector<std::vector<int> > dependencies = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}; // [proc_rank, proc_offset]
	std::vector<bool> dependency_directions;

	first_proc_rank = (backward == true) ? (((m-2)*(n-2)) * (k-1)) % nb_procs : (((m-2)*(n-2)) * (k+1)) % nb_procs;

	// if (!backward && first_proc_rank <= nb_procs) first_proc_rank = 0;

	// if (local_offset % nb_procs >= first_proc_rank) 
	// 	dependencies[Z_DEPENDENCY][0] = (local_offset % nb_procs) - first_proc_rank;
	// else
	// 	dependencies[Z_DEPENDENCY][0] = (local_offset % nb_procs) + first_proc_rank; 

	dependencies[Z_DEPENDENCY][0] = (local_offset + first_proc_rank) % nb_procs;

	dependencies[Z_DEPENDENCY][1] = local_offset;
	dependency_directions = get_dependency_directions(local_offset, m, n);

	if (dependency_directions[LEFT_DEPENDENCY] == true) {
		dependencies[LEFT_DEPENDENCY][0] = (dependencies[Z_DEPENDENCY][0] - 1 < 0) ? nb_procs - 1 : dependencies[Z_DEPENDENCY][0] - 1;
		dependencies[LEFT_DEPENDENCY][1] = local_offset - 1;
	}

	if (dependency_directions[TOP_DEPENDENCY] == true) {
		dependencies[TOP_DEPENDENCY][0] = dependencies[Z_DEPENDENCY][0] + ((m-2) % nb_procs);
		if (dependencies[TOP_DEPENDENCY][0] > nb_procs) { 
			dependencies[TOP_DEPENDENCY][0] = abs(nb_procs - dependencies[TOP_DEPENDENCY][0]);
		} else if (dependencies[TOP_DEPENDENCY][0] == nb_procs) {
			dependencies[TOP_DEPENDENCY][0] = 0;
		}
		dependencies[TOP_DEPENDENCY][1] = local_offset + (m-2);
	}

	if (dependency_directions[RIGHT_DEPENDENCY]) {
		dependencies[RIGHT_DEPENDENCY][0] = (dependencies[Z_DEPENDENCY][0] + 1 >= nb_procs) ? 0 : dependencies[Z_DEPENDENCY][0] + 1;
		dependencies[RIGHT_DEPENDENCY][1] = local_offset + 1;
	}

	if (dependency_directions[BOTTOM_DEPENDENCY]) {
		dependencies[BOTTOM_DEPENDENCY][0] = dependencies[Z_DEPENDENCY][0] - ((m-2) % nb_procs);
		if (dependencies[BOTTOM_DEPENDENCY][0] < 0) {
			dependencies[BOTTOM_DEPENDENCY][0] += nb_procs;
		}
		dependencies[BOTTOM_DEPENDENCY][1] = local_offset - (m-2);
	}

	return dependencies;

}

std::vector<bool> get_dependency_directions(int local_offset, int m, int n) {
	int i, j;
	i = local_offset % (m-2);
	j = local_offset / (m-2);

	std::vector<bool> directions = {true, true, true, true, true};

	if (i == 0) directions[LEFT_DEPENDENCY] = false;
	if (i == (m-2)-1) directions[RIGHT_DEPENDENCY] = false;
	if (j == 0) directions[BOTTOM_DEPENDENCY] = false;
	if (j == (n-2)-1) directions[TOP_DEPENDENCY] = false;

	return directions;
}
