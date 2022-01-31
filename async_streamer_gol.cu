#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#define SRAND_VALUE 12345
#define BLOCK_SIZE 8
 
typedef unsigned char ubyte;

void printBoard(ubyte* grid, int dim){
	for (int i=0; i<dim; i++){
	  for (int j=0; j<dim; j++){
			printf("%c ", grid[i*dim+j]? 'o' : ' ');
		}
		printf("\n");
	}
}

__device__ ubyte getValue(ubyte* grid, int dim, int x, int y){
    if (x >= 0 && x < dim && y >= 0 && y < dim)
        return grid[x * dim + y];
    return 0x0;
}

__device__ int getNeighborsSum(ubyte* grid, int dim, int x, int y){ 
    return getValue(grid, dim, x-1, y-1)
            + getValue(grid, dim, x-1, y)
            + getValue(grid, dim, x-1, y+1)
            + getValue(grid, dim, x, y-1)
            + getValue(grid, dim, x, y+1)
            + getValue(grid, dim, x+1, y-1)
            + getValue(grid, dim, x+1, y)
            + getValue(grid, dim, x+1, y+1);
}

__global__ void simStep(ubyte* grid_curr, ubyte* grid_next, int dim){
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int i = x * dim + y;

    int cell = grid_curr[i];
    int neighbors = getNeighborsSum(grid_curr, dim, x, y);

    if (neighbors < 2)
        grid_next[i] = 0x0;
    else if (neighbors > 3)
        grid_next[i] = 0x0;
    else if (neighbors == 2 || neighbors == 3)
        grid_next[i] = 0x1;
    else
        grid_next[i] = cell;
}

int main(int argc, char* argv[]){
    int gridDim = 256;
    int maxIter = 1000000;
    float probability = 0.2f;
    
    ubyte* h_grid;  
    ubyte* d_grid_curr;
    ubyte* d_grid_next;

    
    size_t boardSize = sizeof(int)*gridDim*(gridDim);
    h_grid = (ubyte*)malloc(boardSize);
    srand(SRAND_VALUE);
    for(int i=0; i<gridDim; i++){
        for(int j=0; j<gridDim; j++){
            h_grid[i*gridDim+j] = (rand() / (float)RAND_MAX <= probability)? 0x1 : 0x0;
        }
    }
    //printBoard(h_grid, gridDim);
    
    clock_t begin = clock();

    cudaMalloc((void **)&d_grid_curr, boardSize);
    cudaMalloc((void **)&d_grid_next, boardSize);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_grid_curr, h_grid, boardSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_grid_next, h_grid, boardSize, cudaMemcpyHostToDevice, stream);

    int gridUnit = (int)ceil(gridDim/(float)BLOCK_SIZE);
    dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridsize(gridUnit, gridUnit, 1);

    ubyte* d_curr;
    ubyte* d_next;
    for (int gen=0; gen<maxIter; gen++){
        if ((gen % 2) == 0){
                d_curr = d_grid_curr;
                d_next = d_grid_next;
            }
            else{
                d_curr = d_grid_next;
                d_next = d_grid_curr;
            }

        simStep<<<gridsize, blocksize, 0, stream>>>(d_curr, d_next, gridDim);
        
        cudaStreamSynchronize(stream);
    }

    cudaMemcpyAsync(h_grid, d_grid_curr, boardSize, cudaMemcpyDeviceToHost, stream);

    cudaStreamDestroy(stream);

    clock_t end = clock();
    double runtime = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%f s", runtime);

    //printf("\n\n");
    //printBoard(h_grid, gridDim);

    return 0;
}
