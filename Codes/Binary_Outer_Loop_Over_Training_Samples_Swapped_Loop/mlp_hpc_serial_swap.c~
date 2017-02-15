/* Code to perform logistic regression - binary classification 
The Code has been changed.
The loops in the double loop structure used for training have been swapped
Here the outer loop goes over all the training samples instead of all the features
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
void adjust_theta(gsl_vector_float* theta, gsl_vector* y,gsl_matrix_float* X, double *partial_grad); 
void predict(gsl_vector_float* theta, gsl_vector* y,gsl_matrix_float* X);
void train(gsl_vector_float* theta, gsl_vector* y,gsl_matrix_float* X);


// important global container i.e. a row of the input matrix
gsl_vector_float_view x_row;
float alpha; //this is called learning rate to update theta vector. This value is found out experimentally
int iteration;


int main (int argc, char* argv[])
{
	char *Xdata = argv[1];
	char *ydata = argv[2];
	int samples_m = atoi(argv[3]);
	int features_n = atoi(argv[4]);
	iteration = atoi(argv[5]);
	alpha = atof(argv[6]);		// size of array
	
	int n;
	double start, end;

	
	FILE * fpx;
	FILE * fpy;

/*	// data set 1
	fpx = fopen ("int_x_dat.txt","r");
	fpy = fopen ("y_data.txt","r");

	gsl_matrix_float* X = gsl_matrix_float_alloc(100, 3);	// matrix to store data along with all its features
	gsl_vector* y = gsl_vector_alloc(100);					// actual answer which is required for training
*/
	// data set 2
	fpx = fopen (Xdata,"r");
	fpy = fopen (ydata,"r");

	gsl_matrix_float* X = gsl_matrix_float_alloc(samples_m, features_n);
	gsl_vector* y = gsl_vector_alloc(samples_m);
	
	gsl_matrix_float_fscanf (fpx, X);
	gsl_vector_fscanf (fpy, y);

	fclose(fpx);
	fclose(fpy);

	// n = no. of features to perform classification
	n = (X->size2)-1; // excluding intercept term, for first ex. n = 3-1 = 2

	// initialize theta
	gsl_vector_float* theta = gsl_vector_float_alloc(n+1);	// theta = learned values/weights corresponding to all features n
	gsl_vector_float_set_zero(theta) ; // initialize theta vector to zero


	// trainining based on input data
	start= omp_get_wtime();
	train(theta, y, X); 
	end = omp_get_wtime();  

	// time take to learn weights in vector theta
	printf("time: %f\n",end-start);


/* CODE FOR CHECKING AND TESTING */

	// check
	// print the theta obtained
/*	for(ite = 0; ite < n; ite++)
	{
		printf("%f ", gsl_vector_float_get(theta,ite));
		printf("\n");
	}
*/

	// test
	// cost after learning must be as small as possible
//	printf("cost after learning: %f\n", get_cost(theta, y, X));


/*	//testing the output
	//predict
	// check test case 1 45 85
	gsl_vector_float* test = gsl_vector_float_alloc(3);

	gsl_vector_float_set (test,0,1);
	gsl_vector_float_set (test,1,45);
	gsl_vector_float_set (test,2,85);   

	gsl_vector_float_set (theta,0,-19.414051);
	gsl_vector_float_set (theta,1,0.160308);
	gsl_vector_float_set (theta,2,0.154952);   

	//predict(theta,y,X);
	//printf("prob_pred:%f\n",sigmoid(test,theta));

	//gsl_vector_float_free (a);
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
	// Currently this function is included in the train itself to reduce function call overhead.
*/
void adjust_theta(gsl_vector_float* theta, gsl_vector* y, gsl_matrix_float* X, double *partial_grad) // grad vector now used now
{  
	int n = X->size2, m = X->size1, j, i;

//	float alpha = 0.002; //this is called learning rate to update theta vector. This value is found out experimentally

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
		double sig_row_y_i = sigmoid(&x_viewrow.vector,theta) - gsl_vector_get(y,i);

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
		gsl_vector_float_set (theta, j, gsl_vector_float_get(theta, j)-(alpha*(partial_grad[j]/m)) );
	}
}



/* This function actualy performs the training over the given data
	INPUT: theta vector, y vector and input X matrix
	OUTPUT: Void
*/
void train(gsl_vector_float* theta, gsl_vector* y, gsl_matrix_float* X)
{
	int ite;
	//iteration = 100;		// iterating to obtain theta vector as accurate as possible
  
  	int n = X->size2, m = X->size1, j, i;

//	float alpha = 0.1; //this is called learning rate to update theta vector. This value is found out experimentally

	gsl_vector_float_view x_viewrow;

	double partial_grad[n];

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
			double sig_row_y_i = sigmoid(&x_viewrow.vector,theta) - gsl_vector_get(y,i);

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
			gsl_vector_float_set (theta, j, gsl_vector_float_get(theta, j)-(alpha*(partial_grad[j]/m)) );
		}  
	}

/*
	// for testing purpose
	// IO PORTION FOR PREDICT
  
  	// read the testing files here
	FILE * fpx;
	FILE * fpy;
	fpx = fopen ("X_bin_test.dat","r");
	fpy = fopen ("y_bin_test.dat","r");

	gsl_matrix_float* Xtest = gsl_matrix_float_alloc(299,401);
	gsl_vector* ytest = gsl_vector_alloc(299);

	gsl_matrix_float_fscanf (fpx,Xtest);
	gsl_vector_fscanf (fpy,ytest);

	fclose(fpx);
	fclose(fpy);


	printf("now predicting %f\n", (float)y -> size);
	predict(theta,y,X);
*/
}



/* Function to predict/classify the given input. Here it is a binary classification therefore prediction would be either 0 or 1
	INPUT: trained theta vector, y vector, input X matrix
	OUTPUT: Void
*/
void predict(gsl_vector_float* theta, gsl_vector* y, gsl_matrix_float* X)
{
    int i, m = X->size1;
    
    int correct = 0;		// counter to count the correct predictions made by the trained model
    
    double sig; 
    
    for (i = 0; i < m; i++)
    {
		x_row = gsl_matrix_float_row(X, i);	// an input is taken in x_row for prediction 
		sig = sigmoid(&x_row.vector, theta);		// find sigmoid value for the given input stored in x_row

		// Since this is a binary classification, the threshold is taken as 0.5
		// All sig values >= 0.5 are considered 1 and sig values < 0.5 are considered 0.
		// The above conversion is done by the formula -> floor(2*sig)
		if ( gsl_vector_get(y, i) == floor(2*sig) )		// checking the above predicted value with the actual value stored in y vector
			correct++;								// if the value matches, it is a correct prediction
    }
    
    // find accuracy in prediction
    printf("accuracy: %f\n", correct/(float)y -> size );
}
