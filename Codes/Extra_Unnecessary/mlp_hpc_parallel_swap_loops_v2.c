#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include <omp.h>

float sigmoid(gsl_vector_float* x, gsl_vector_float* theta);
double get_cost(gsl_vector_float* theta, gsl_vector* y,gsl_matrix_float* X);
float get_gradient(gsl_vector_float* theta, gsl_vector* y,gsl_matrix_float* X,gsl_vector_float* grad);
void adjust_theta(gsl_vector_float* theta, gsl_vector* y,gsl_matrix_float* X); 
void predict(gsl_vector_float* theta, gsl_vector* y,gsl_matrix_float* X);
void train(gsl_vector_float* theta, gsl_vector* y,gsl_matrix_float* X);


// important global container i.e. a row of the input matrix
gsl_vector_float* x_row;



int main (void)
{
	int n;
	double start, end;

	// data set 1
	FILE * fpx;
	FILE * fpy;

	fpx = fopen ("int_x_dat.txt","r");
	fpy = fopen ("y_data.txt","r");

	gsl_matrix_float* X = gsl_matrix_float_alloc(100, 3);	// matrix to store data along with all its features
	gsl_vector* y = gsl_vector_alloc(100);					// actual answer which is required for training

	// data set 2
/*	fpx = fopen ("X_bin_update.dat","r");
	fpy = fopen ("y_bin.dat","r");

	gsl_matrix_float* X = gsl_matrix_float_alloc(1499, 401);
	gsl_vector* y = gsl_vector_alloc(1499);
*/
	gsl_matrix_float_fscanf (fpx, X);
	gsl_vector_fscanf (fpy, y);

	fclose(fpx);
	fclose(fpy);

	// n = no. of features to perform classification
	n = (X->size2)-1; // excluding intercept term, for first ex. n = 3-1 = 2

	// initialize theta
	gsl_vector_float* theta = gsl_vector_float_alloc(n+1);	// theta = learned values/weights corresponding to all features n
	gsl_vector_float_set_zero(theta) ; // initialize theta vector to zero


	// allocate the vectors needed in the adjust theta function, maybe needed for calculate cost as well
	x_row = gsl_vector_float_alloc(X->size2);		// size of row of X = no. of features = n

	// trainining based on input data
	start= omp_get_wtime();
	train(theta, y, X); 
	end = omp_get_wtime();  

	// time take to learn weights in vector theta
	printf("time: %f\n",end-start);


	// check
	// print the theta obtained
/*	for(ite = 0; ite < n; ite++)
	{
		printf("%f ", gsl_vector_float_get(theta,ite));
		printf("\n");
	}
*/


	// cost after learning must be as small as possible
	printf("cost after learning: %f\n", get_cost(theta, y, X));


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
    
    gsl_vector_float* x_row = gsl_vector_float_alloc(X->size2);	
    
    double partial_cost = 0.0, hyp;		// hyp means hypothesis
    
    for (i = 0; i < m; i++)
    {
        x_row = gsl_matrix_float_get_row(X, i);	// Get i'th row of X (i'th feature) and copy in vector
        y_i = gsl_vector_get(y, i);				// get i'th element of result vector y
        hyp = sigmoid(x_row, theta);			// find sigmoid of that feature	
          
        partial_cost += ( ( y_i*log(hyp) ) + (1-y_i)*log(1- hyp));	// the cost is calculated over all the features using hypothesis value 
    }
    
    return (-1.0/m)*partial_cost;   	// returns total cost which is a scalar
}



/* This function finds the best possible parameters of vector theta such that the cost function which is the cost incurred by the programme due to a wrong prediction is minimized.
	INPUT: theta vector, y vector, input X matrix
	OUTPUT: Void
*/

// seperate func for grad calc for now, can be comb with cost

// this is one iteration of the gradient descent
// run this for required number of iterations


void adjust_theta(gsl_vector_float* theta, gsl_vector* y, gsl_matrix_float* X) // grad vector now used now
{  
	int n = X->size2, m = X->size1, j;// i;	// for our example n = 3, m = 100

	float lambda = 0.002; //this is called learning rate to update theta vector. try different values, 0.001 learning rate okay but slow, 0.002

	gsl_vector_float_view X_j, x_viewrow;
	
	//gsl_vector_float* X_j; // make this global so that not have to init everytime in func
	///gsl_vector_float* x_row; // make global

	double partial_grad=0.0; // where should par grad be made to zero,
	
	// as collapse is not used. only outer loop is parallel
	# pragma omp parallel for private (X_j, j, x_viewrow, partial_grad) 
	for (j = 0; j < n ;j++)
	{
		int i; //used when collapse level; is 1
		
		partial_grad=0;
		
		// init X_j otherwise get core dumped
		//gsl_matrix_float_get_col(X_j,X,j);
	 
		X_j = gsl_matrix_float_column(X, j);

		// loop which calculates gradient of cost function for given theta vector
		for (i = 0; i < m; i++)
		{
			x_viewrow=gsl_matrix_float_row(X,i);
			//gsl_matrix_float_get_row(x_row, X, i);		// get i'th row of matrix X
			partial_grad += ( (sigmoid(&x_viewrow.vector,theta) - gsl_vector_get(y,i) )* gsl_vector_float_get(&X_j.vector,i) ); 
			//partial_grad += ( ( sigmoid(x_row, theta) - gsl_vector_get(y, i) ) * gsl_vector_float_get(X_j, i) );   
		}
	
		//gsl_vector_float_set (grad,j,(partial_grad/m)) ;

		// set weight acc grad des rule

		// set j th theta element. Update based on the calculation of partial gradient to minimize cost
		gsl_vector_float_set (theta, j, gsl_vector_float_get(theta, j)-(lambda*(partial_grad/m)) ); // theta(j) = theta(j) - lambda * [partial derivative of cost function w.r.t. theta(j)]

		//debug compute cost for every iteration
		//  printf("%f ",get_cost(theta,y,X));     
	}

	// update weights here itself, use gradient descent
}



/* This function actualy performs the training over the given data
	INPUT: theta vector, y vector and input X matrix
	OUTPUT: Void
*/
void train(gsl_vector_float* theta, gsl_vector* y, gsl_matrix_float* X)
{
	int ite, iteration=1000000;		// iterating 1e6 times to obtain theta vector as accurate as possible
  
  	int n = X->size2, m = X->size1, i;	// for our example n = 3, m = 100

	float lambda = 0.002; 			//this is called learning rate to update theta vector.

	gsl_vector_float_view X_j1, X_j2, X_j3, x_viewrow;
  
  	X_j1 = gsl_matrix_float_column(X, 0);
	X_j2 = gsl_matrix_float_column(X, 1);
	X_j3 = gsl_matrix_float_column(X, 2);
	
	double partial_grad1=0.0, partial_grad2=0.0, partial_grad3=0.0;

	for (ite = 1; ite <= iteration; ite++)
	{
//		adjust_theta(theta, y, X); //function to adjust theta according to the data samples

		partial_grad1=0.0, partial_grad2=0.0, partial_grad3=0.0;
	
		// as collapse is not used. only outer loop is parallel
		
//		for (j = 0; j < n ;j++)
//		{
//			int i; //used when collapse level; is 1
		
	//		partial_grad=0;
		
			// init X_j otherwise get core dumped
			//gsl_matrix_float_get_col(X_j,X,j);
		 
			
			// loop which calculates gradient of cost function for given theta vector
			# pragma omp parallel for reduction(+:partial_grad1,partial_grad2,partial_grad3) 
			for (i = 0; i < m; i++)
			{
//				printf("%d ", i);
				x_viewrow=gsl_matrix_float_row(X,i);
				double sig_row_y_i = sigmoid(&x_viewrow.vector,theta) - gsl_vector_get(y,i);
				
				partial_grad1 += ( sig_row_y_i * gsl_vector_float_get(&x_viewrow.vector,0) );
				partial_grad2 += ( sig_row_y_i * gsl_vector_float_get(&x_viewrow.vector,1) );
				partial_grad3 += ( sig_row_y_i * gsl_vector_float_get(&x_viewrow.vector,2) );    
			}

//			printf("%d: %f, %f, %f\n", ite, partial_grad1, partial_grad2, partial_grad3);
			
			// set j th theta element. Update based on the calculation of partial gradient to minimize cost
			gsl_vector_float_set (theta, 0, gsl_vector_float_get(theta, 0)-(lambda*(partial_grad1/m)) ); // theta(j) = theta(j) - lambda * [partial derivative of cost function w.r.t. theta(j)]    
			gsl_vector_float_set (theta, 1, gsl_vector_float_get(theta, 1)-(lambda*(partial_grad2/m)) );
			gsl_vector_float_set (theta, 2, gsl_vector_float_get(theta, 2)-(lambda*(partial_grad3/m)) );
//		}
	}
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
		sig = sigmoid(x_row, theta);		// find sigmoid value for the given input stored in x_row

		// Since this is a binary classification, the threshold is taken as 0.5
		// All sig values >= 0.5 are considered 1 and sig values < 0.5 are considered 0.
		// The above conversion is done by the formula -> floor(2*sig)
		if ( gsl_vector_get(y, i) == floor(2*sig) )		// checking the above predicted value with the actual value stored in y vector
			correct++;								// if the value matches, it is a correct prediction
    }
    
    // find accuracy in prediction
    printf("accuracy: %f\n", correct/(float)y -> size );
}
