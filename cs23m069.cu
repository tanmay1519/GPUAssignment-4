#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <math.h>
using namespace std;

//*******************************************

// Write down the kernels here


//***********************************************



__global__ void clarity (int *A,int n,int val){
  int  id = blockDim.x*blockIdx.x +threadIdx.x ;
  if(id<n){
    A[id]= val ;
  }
}






__global__ void nearest(int *health ,int *kx,int *ky ,int *round,int T,int *kdistance,int *kscore , int *newhealth){


  
   int tank =blockIdx.x;
   int i = round[0];
   if (i%T!=0){
    int currenttank =   threadIdx.x;
    if (currenttank==0)
    kdistance[tank]=INT_MAX;

__syncthreads();
    int actual = (tank+i)%T ;
   int x1,x2,y1,y2,x3,y3;
   int d1 = -1,a,b,a1,b1,mycheck ;
   
if (tank!=currenttank&&health[tank]>0&&health[currenttank]>0){

    x1=kx[tank];
    y1=ky[tank];
    x2=kx[currenttank];
    y2=ky[currenttank];
    x3=kx[actual];
    y3=ky[actual];
    

     a = y3-y1;b=x3-x1;
     a1 = y2-y1 ;
     b1 = x2-x1;
    mycheck = 1 ;
  if ((x1>x2&&x1<x3)||(x1<x2&&x1>x3)||(y1>y2&&y1<y3)||(y1<y2&&y1>y3)){
            mycheck = 0;
        }
 



  //  __syncthreads();

  if((( a*b1 )== (b*a1)) && mycheck&&health[tank]>0&&health[currenttank]>0 ){
int c,d ;
c=abs(a1);
d=abs(b1);
    // d1 = c*c+d*d    ;
    d1=c+d;
        // printf("%d %d %d %d\n",tank,currenttank,d1,kdistance[tank]);
       

                atomicMin(&kdistance[tank],d1);

            }
    


    __syncthreads();
    
    if (d1==kdistance[tank]){

    
        atomicSub(&newhealth[currenttank],1);
        atomicAdd(&kscore[tank],1);

    }
}}
    }






__global__ void copypaste (int *A,int *B ){
  int id = threadIdx.x;
  A[id]=B[id];
}


__global__ void checker(int *khealth,int *check,int *kscore,int *round){
    int tankid = threadIdx.x;
    if(tankid==0) atomicAdd(&round[0],1);
    if (khealth[tankid]>0) atomicAdd(&check[0],1);
   
}

int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int *kx,*ky,*khealth,*check,*kscore,*knewhealth ,*kdistance;

    int mycheck[1];
    mycheck[0]=T;// tryb pointer
   

    cudaMalloc(&kx,T*(sizeof(int)));

    cudaMalloc(&ky,T*(sizeof(int)));
    cudaMalloc(&kdistance,T*(sizeof(int)));



    cudaMalloc(&khealth,T*(sizeof(int)));
    cudaMalloc(&knewhealth,T*(sizeof(int)));

    cudaMalloc(&kscore,T*(sizeof(int)));

    cudaMalloc(&check,(sizeof(int)));
   
        clarity<<<1,1>>>(check,1,1);

    

    cudaMemcpy(kx,xcoord,T*(sizeof(int)),cudaMemcpyHostToDevice);
    cudaMemcpy(ky,ycoord,T*(sizeof(int)),cudaMemcpyHostToDevice);

    clarity<<<1,T>>> (khealth,T,H);
    clarity<<<1,T>>> (knewhealth,T,H);
    int *round ;
    cudaMalloc(&round,(sizeof(int)));
    clarity<<<1,1>>>(round,1,1);
  
 
    while (mycheck[0]>1){
     
       
       
        nearest<<<T,T>>>(khealth,kx,ky,round,T,kdistance,kscore,knewhealth);
     
       copypaste<<<1,T>>>(khealth,knewhealth);
        clarity<<<1,1>>>(check,1,0);
        checker<<<1,T>>>(khealth,check,kscore,round);
        cudaDeviceSynchronize();
        cudaMemcpy(mycheck,check,sizeof(int),cudaMemcpyDeviceToHost);
  
        
    }


cudaMemcpy(score,kscore,T*sizeof(int),cudaMemcpyDeviceToHost);


    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}