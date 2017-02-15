/* Code to perform logistic regression - Multiclass classification 

This code is an extension of the binary classification code
It performs training for classification over multiple classes
*/

#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include <omp.h>

float sigmoid(gsl_vector_float* x, gsl_vector_float* theta);
double get_cost(gsl_vector_float* theta, gsl_vector* y,gsl_matrix_float* X);
float get_gradient(gsl_vector_float* theta, gsl_vector* y,gsl_matrix_float* X,gsl_vector_float* grad);
void adjust_theta(gsl_vector_float_view theta, gsl_vector* y,gsl_matrix_float* X, double *partial_grad); 
void predict(gsl_matrix_float* thetamat, gsl_vector* y,gsl_matrix_float* X);
void train(gsl_matrix_float* thetamat, gsl_vector* y,gsl_matrix_float* X);


// important global container i.e. a row of the input matrix
gsl_vector_float_view x_row;
float alpha; //this is called learning rate to update theta vector. This value is found out experimentally
int iteration;
int k;

int main (int argc, char* argv[])
{
	char *Xdata = argv[1];
	char *ydata = argv[2];
	int samples_m = atoi(argv[3]);
	int features_n = atoi(argv[4]);
	iteration = atoi(argv[5]);
	alpha = atof(argv[6]);		// size of array
	k = atoi(argv[7]);				// k is the number of classes in which we have to distinguish


//	printf("%s %s %d %d %d %f\n", Xdata, ydata, samples_m, features_n, iteration, alpha);


	int n;
	double start, end;


	// read data from file
	FILE * fpx;
	FILE * fpy;
	fpx = fopen (Xdata,"r");
	fpy = fopen (ydata,"r");

	gsl_matrix_float* X = gsl_matrix_float_alloc(samples_m,features_n);
	gsl_vector* y = gsl_vector_alloc(samples_m); // 4000 train and 1000 test


	gsl_matrix_float_fscanf (fpx,X);
	gsl_vector_fscanf (fpy,y);


	fclose(fpx);
	fclose(fpy);
  
  
    
	// k is the number of classes in which we have to distinguish
//	k = 9; // for this input data
  
	// n = no. of features to perform classification
	n = (X->size2)-1; // excluding intercept term, for first ex. n = 3-1 = 2
  
	// theta is now matrix
	gsl_matrix_float* thetamat = gsl_matrix_float_alloc(k,n+1);
	gsl_matrix_float_set_zero (thetamat) ; // set to zero

	// trainining based on input data
	start= omp_get_wtime();
	train(thetamat, y, X); 
	end = omp_get_wtime();  

	// time take to learn weights in vector theta
	printf("time: %f\n",end-start);


/* CODE FOR CHECKING AND TESTING */
/*
	// cost after learning must be as small as possible
	printf("cost after learning: %f\n", get_cost(theta, y, X));
*/
	return 0;
}



/* This function is used as the logistic regression hypothesis function 
	INPUT: theta vector and x vector
	OUTPUT: hypothesis vector of x for given theta vector
*/
float sigmoid(gsl_vector_float* x, gsl_vector_float* theta)
{
   float result;
   gsl_blas_sdot(theta, x, &result);	// theta-transpose * x is essentially dot product of theta and x which is a scalar float value stored in result.
   // NOTE: Here we are not parallelizing this scalar product because the number of features taken for prediction in any practical situation is of the order of 1e4 which is not sufficient to get better result in parallel implementation.
    
   return 1.0/(1+exp(-1*result));
}



/* function to calculate the cost parameter for logical regression. This is the cost incurred if the prediction is wrong
	INPUT: Theta vector, input X matrix and y vector
	OUTPUT: The cost value J(theta) which is scalar.	
	
	This function is used for checking the total cost that will be incurred at any point during the training
*/
double get_cost(gsl_vector_float* theta, gsl_vector* y, gsl_matrix_float* X)
{
    int i, m = X->size1, y_i;
    
    double partial_cost = 0.0, hyp;		// hyp means hypothesis
    
    for (i = 0; i < m; i++)
    {
        x_row = gsl_matrix_float_row(X, i);	// Get i'th row of X (i'th feature) and copy in vector
        y_i = gsl_vector_get(y, i);				// get i'th element of result vector y
        hyp = sigmoid(&x_row.vector, theta);			// find sigmoid of that feature	
          
        partial_cost += ( ( y_i*log(hyp) ) + (1-y_i)*log(1- hyp));	// the cost is calculated over all the features using hypothesis value 
    }
    
    return (-1.0/m)*partial_cost;   	// returns total cost which is a scalar
}




/* This function finds the best possible parameters of vector theta such that the cost function which is the cost incurred by the programme due to a wrong prediction is minimized.
	INPUT: theta vector, y vector, input X matrix
	OUTPUT: Void
	
	// this is one iteration of the gradient descent
	// run this for required number of iterations
	Currently this function is included in the train itself to reduce function call overhead.
*/
void adjust_theta(gsl_vector_float_view theta, gsl_vector* y, gsl_matrix_float* X, double *partial_grad) // grad vector now used now
{  
	int n = X->size2, m = X->size1, j, i;

//	alpha = 0.002; //this is called learning rate to update theta vector. This value is found out experimentally

	gsl_vector_float_view x_viewrow;

//	double partial_grad[n];
	
	// make all the values of partial gradient vector zero
	for (j = 0; j < n ;j++)
	{
		partial_grad[j] = 0.0;
	}

	// iterate over all the data samples
	for (i = 0; i < m; i++)
	{
		x_viewrow = gsl_matrix_float_row(X,i);
	
		// calculate sigmoid
		//NOTE: Here we need to calculate sigmoid only once for all the features of that smaple as opposed to each time in the previous implementation
		// This reduces the work complexity and time required
		double sig_row_y_i = sigmoid(&x_viewrow.vector,&theta.vector) - gsl_vector_get(y,i);

		// for each column i.e. for each feature, calculate partial gradient and add it to appropriate vector element
		for (j = 0; j < n ;j++)
		{
			partial_grad[j] += ( sig_row_y_i * gsl_vector_float_get(&x_viewrow.vector,j) );
		}   
	}
	
	// update all values in theta vector based on corresponding calculated values in partial_grad gradient vector
	for (j = 0; j < n ;j++)
	{
		//gradient descent rule: theta(j) = theta(j) - alpha * [partial derivative of cost function w.r.t. theta(j)]
		// set j th theta element. Update based on the calculation of partial gradient to minimize cost
		gsl_vector_float_set (&theta.vector, j, gsl_vector_float_get(&theta.vector, j)-(alpha*(partial_grad[j]/m)) );
	}
}



/* This function actualy performs the training over the given data
	INPUT: theta vector, y vector and input X matrix
	OUTPUT: Void
*/
void train(gsl_matrix_float* thetamat, gsl_vector* y, gsl_matrix_float* X)
{
	int ite;
	// iteration=100;
  
  	int n = X->size2, m = X->size1, i, j;	// for our example n = 3, m = 100

	int class,size;
  
	gsl_vector* ytemp; 				// temporary vector y to perform training using binary classification
	size = y->size;
	ytemp = gsl_vector_alloc(size);

//	alpha = 0.1; 			//this is called learning rate to update theta vector. This value is found out experimentally

	gsl_vector_float_view x_viewrow;

	double partial_grad[n];
	
	// put a loop here to train for k classes, create theta matrix and process y as belonging to kth class or not
	for(class = 1; class < 10; class++)
	{
		// class represents current class on which we train

		//create row view of thetamat and process y for use with k

		// ytemp is processed according to class k i.e. for class k elements, it is set to 1 while for other class elements, it is set to 0
		// so that we can perform binary classification
		for(i=0;i<m;i++)
		{
			if (gsl_vector_get(y,i)==class)
				gsl_vector_set(ytemp,i,1);
			else
				gsl_vector_set(ytemp,i,0);
				
		}     

		// create a view for theta of thetamat row
		gsl_vector_float_view theta; // a row of the thetamat
		theta = gsl_matrix_float_row (thetamat,class-1); // 0 indexed

	
		// this loop must run sequentially due to data dependency. It cannot be parallelized
		for (ite = 1; ite <= iteration; ite++)
		{
	//		adjust_theta(theta, y, X, partial_grad); //function to adjust theta according to the data samples


			// make all the values of partial gradient vector zero
			for (j = 0; j < n ;j++)
			{
				partial_grad[j] = 0.0;
			}
	
			// iterate over all the data samples
			for (i = 0; i < m; i++)
			{
				x_viewrow = gsl_matrix_float_row(X,i);
			
				// calculate sigmoid
				//NOTE: Here we need to calculate sigmoid only once for all the features of that smaple as opposed to each time in the previous implementation
				// This reduces the work complexity and time required
				double sig_row_y_i = sigmoid(&x_viewrow.vector,&theta.vector) - gsl_vector_get(y,i);

				// for each column i.e. for each feature, calculate partial gradient and add it to appropriate vector element
				for (j = 0; j < n ;j++)
				{
					partial_grad[j] += ( sig_row_y_i * gsl_vector_float_get(&x_viewrow.vector,j) );
				}   
			}
			
			// update all values in theta vector based on corresponding calculated values in partial_grad gradient vector
			for (j = 0; j < n ;j++)
			{
				//gradient descent rule: theta(j) = theta(j) - alpha * [partial derivative of cost function w.r.t. theta(j)]
				// set j th theta element. Update based on the calculation of partial gradient to minimize cost
				gsl_vector_float_set (&theta.vector, j, gsl_vector_float_get(&theta.vector, j)-(alpha*(partial_grad[j]/m)) );
			}  
		}
	
//check		printf("class %d done\n",class);
	
	}
	

/* CODE FOR CHECKING AND TESTING */
	
	/* IO PORTION FOR PREDICT*/
/*  
  FILE * fpx;
  FILE * fpy;
  fpx = fopen ("X_multi_test.dat","r"); //will have to use diff set
  fpy = fopen ("y_multi_test.dat","r");
 
  gsl_matrix_float* Xtest = gsl_matrix_float_alloc(1000,401);
  gsl_vector* ytest = gsl_vector_alloc(1000);
  
  
  gsl_matrix_float_fscanf (fpx,Xtest);
  gsl_vector_fscanf (fpy,ytest);
  
  
  fclose(fpx);
  fclose(fpy);
 */ 
  
  // read the testing files here
  
  
//  printf("now predicting %f\n", (float)ytest -> size);
//  predict(thetamat,ytest,Xtest);
}



/* Function to predict/classify the given input. Here it is a multivalue classification therefore prediction would be any value from 1 to 9
	INPUT: trained theta vector, y vector, input X matrix
	OUTPUT: Void
*/
void predict(gsl_matrix_float* thetamat, gsl_vector* y,gsl_matrix_float* X)
{
    int i, m = X->size1, class, maxclass;
    
    int correct = 0;		// counter to count the correct predictions made by the trained model
    
    gsl_vector_float_view theta;	// to store theta vector of a particular class
     
    double sig, maxprob = 0.0; 
    
    // iterate over each sample that is to be predicted
    for (i = 0; i < m; i++)
    {
		maxprob=0.0;		// will hold the maximum probability 
		x_row = gsl_matrix_float_row(X,i);	// get i'th sample

		// iterate over all classes
		// a for loop here, find the k for which we get max hyp, use typical max code.
		for(class = 1; class < 10; class++)
		{
			theta = gsl_matrix_float_row(thetamat, class-1);  
		
			// find the sigmoid value over all classes
			sig = sigmoid(&x_row.vector,&theta.vector);
		
			// take the maximum sigmoid value
			if (maxprob < sig)
			{
				maxprob = sig;
				maxclass = class;	// class with maximum sigmoid value is the maxclass
			}       
		} 

		// Now maxclass is the most probable class to which the sample belongs
		
		
		// check with the actual answer which we have
		if ( gsl_vector_get(y,i) == maxclass )		// if it matches, then the prediction is correct
			correct++;
    }
    
    printf("accuracy:%f\n",correct/(float)(y->size));
}
