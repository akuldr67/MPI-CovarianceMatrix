#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

//returns the value to placed in covarience matrix, given 2 arrays
double findCov(double x1[], double x2[], int n){
	double ans = 0;
	for(int i=0;i<n;i++){
		ans+=(x1[i]*x2[i]);
	}
	return ans;
}

void main(int argc, char *argv[]){
	FILE *fp = fopen("iris.data","r");
	const char s[2] = ",";
	char *token;
	int i,j;
	int count=0;
	int rows=0;
	double elapsed_time;
	int numprocs, myid,Root=0;
	int remaining_first,local_n,my_first,remaining_length,total_local;

	/*....MPI Initialisation....*/
	MPI_Init(&argc, &argv);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	if(myid == Root){
		if(fp!=NULL){
			for (char c = getc(fp); c != EOF; c = getc(fp)){
	        	if (c == '\n')
	            	count = count + 1; 
	    	}
		}
		fclose(fp);
		fp = fopen("iris.data","r");
		char line[256];
		if(fgets(line, sizeof line, fp)!=NULL){
			token = strtok(line,s);
			i=0;
			while(token!=NULL){
				i++;
				token = strtok(NULL,s);
			}
			rows=i-1;
		}
	}
	fclose(fp);

	MPI_Bcast(&count, 1, MPI_INT, Root, MPI_COMM_WORLD);
	MPI_Bcast(&rows, 1, MPI_INT, Root, MPI_COMM_WORLD);

	double data[count][rows];
	double sum[rows];

	int countIterator = 0;


	fp = fopen("iris.data","r");

	if(fp==NULL)
		printf("can't open dataset\n");
	else{
		//reading data
		if(myid == Root){
			char line[256];
			while(fgets(line, sizeof line, fp)!=NULL){
				//getting token by dividing line over every ','
				token = strtok(line,s);
				
				if(strcmp(line,"\n")!=0){

					i=0;
					//while there are more tokens present in that line
					while(1){
						//getting next token, so that last column could be ignored
						char *token2 = token;
						token = strtok(NULL,s);
						if(token!=NULL){
							float x = atof(token2);
							data[countIterator][i] = x; 	//storing data
							// sum[i]+=x;				//calculating sum for mean
						}
						else break;
						i++;
					}
					//rows will store the number of columns
					// if(count==0)
					// 	rows=i;
					//counting no of rows (data enteries)
					countIterator++;

				}
			}
		}
		fclose(fp);

		//starting the timer
		elapsed_time = - MPI_Wtime();


		int sendCounts[numprocs];
		int displs[numprocs];
		int sendCounts2[numprocs];
		int displs2[numprocs];
		local_n = count/numprocs;
		int rem = count%numprocs;

		//dividing data to processes
		displs[0] = 0;
		if(rem>0) sendCounts[0] = local_n + 1;
		else sendCounts[0] = local_n;

		for(i=1;i<numprocs;i++){
			displs[i] = displs[i-1]+sendCounts[i-1];
			if(i<rem){
				sendCounts[i] = local_n+1;
			}
			else sendCounts[i] = local_n;
		}

		for(i=0;i<numprocs;i++){
			sendCounts2[i] = sendCounts[i];
			displs2[i] = displs[i];
			sendCounts[i]*=rows;
			displs[i]*=rows;
		}

		double local_data[sendCounts2[myid]][rows];

		// calling MPI_Scatterv
		MPI_Scatterv(data, sendCounts, displs, MPI_DOUBLE, local_data, sendCounts[myid], MPI_DOUBLE, Root, MPI_COMM_WORLD);


		double local_sum[rows];

		//calculating local sum of local data
		for(i=0;i<rows;i++){
			local_sum[i]=0;
		}
		for(i=0;i<sendCounts2[myid];i++){
			for(j=0;j<rows;j++)
				local_sum[j]+=local_data[i][j];
		}

		//reducing local sum to global sum
		MPI_Reduce(&local_sum, &sum, rows, MPI_DOUBLE, MPI_SUM, Root, MPI_COMM_WORLD);


		//calculating mean in root process
		double mean[rows];
		if(myid==Root){
			for(j=0;j<rows;j++){
				mean[j]=sum[j]/(count*1.0);
			}
		}

		//broadcasting mean to all process
		MPI_Bcast(&mean, rows, MPI_DOUBLE, Root, MPI_COMM_WORLD);


		// printing mean in root process
		if(myid==Root){
			printf(" Mean vector: [ ");
			for(i=0;i<rows;i++){
				if(i==rows-1)
					printf("%.2f ] \n", mean[i]);
				else printf("%.2f , ", mean[i]);
			}
		}


		//making local data for covariance matrix calculation
		for(i=0;i<sendCounts2[myid];i++){
			for(j=0;j<rows;j++)
				local_data[i][j]-=mean[j];
		}


		double local_data2[rows][sendCounts2[myid]];
		double cov[rows][rows];
		double local_cov[rows][rows];

		//local data2 will store the transpose of the data
		for(j=0;j<sendCounts2[myid];j++){
			for(i=0;i<rows;i++)
				local_data2[i][j] = local_data[j][i];
		}

		//calculating local covariance
		for(i=0;i<rows;i++){
			for(j=0;j<rows;j++){
				local_cov[i][j] = findCov(local_data2[i],local_data2[j],sendCounts2[myid]);
			}
		}

		//reducing local cov to global cov
		MPI_Reduce(&local_cov, &cov, rows*rows, MPI_DOUBLE, MPI_SUM, Root, MPI_COMM_WORLD);

		//printing and calculating cov in root process
		if(myid==Root){
			printf("\n Covarience matrix: \n");
			for(i=0;i<rows;i++){
				for(j=0;j<rows;j++){
					cov[i][j] /= (count-1);
					printf("%.3lf ", cov[i][j]);
				}
				printf("\n");
			}
			elapsed_time+=MPI_Wtime();
			printf ("\nTime taken = %f\n", elapsed_time);
		}

	}
	MPI_Finalize();
}
