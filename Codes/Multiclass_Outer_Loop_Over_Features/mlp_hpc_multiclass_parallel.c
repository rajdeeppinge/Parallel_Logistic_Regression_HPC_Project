/* This is the multiclass implementation of the slower method just for the sake of checking 
	This is the parallel version to check if we obtain linear speedup
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
void adjust_theta(gsl_vector_float_view theta, gsl_vector* y,gsl_matrix_float* X); // grad vec not used now
void predict(gsl_matrix_float* thetamat, gsl_vector* y,gsl_matrix_float* X);
void train(gsl_matrix_float* thetamat, gsl_vector* y,gsl_matrix_float* X);


// important global container i.e. a row of the input matrix
gsl_vector_float_view x_row;



int main (void)
{
	int n, k;
	double start, end;


	// read data from file
	FILE * fpx;
	FILE * fpy;
	fpx = fopen ("X_multi_input.dat","r");
	fpy = fopen ("y_multi_input.dat","r");

	gsl_matrix_float* X = gsl_matrix_float_alloc(4000,401);
	gsl_vector* y = gsl_vector_alloc(4000); // 4000 train and 1000 test


	gsl_matrix_float_fscanf (fpx,X);
	gsl_vector_fscanf (fpy,y);


	fclose(fpx);
	fclose(fpy);
  
  
    
	// k is the number of classes in which we have to distinguish
	k = 9; // for this input data
  
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
*/
void adjust_theta(gsl_vector_float_view theta, gsl_vector* y,gsl_matrix_float* X) // grad vector now used now
{
	int n = X->size2, m = X->size1, j, i;

	float lambda = 0.1; //this is called learning rate to update theta vector. This value is found out experimentally

	gsl_vector_float_view x_viewrow;
	
	double partial_grad = 0.0;
   
   	# pragma omp parallel for private (i,j,x_viewrow,partial_grad) 
	for (j = 0; j < n ;j++)		// run loop over all features i.e. n
	{
		partial_grad = 0.0;	// initialize partial_gradient for every feature

		// loop which calculates gradient of cost function for given theta vector
		for (i = 0; i < m; i++)
		{
			x_viewrow = gsl_matrix_float_row(X,i);		// get i'th row of matrix X
		
			// calculate partial gradient for given feature over whole data
			partial_grad += ( ( sigmoid(&x_viewrow.vector, &theta.vector) - gsl_vector_get(y, i) ) * gsl_vector_float_get(&x_viewrow.vector, j) );   
		}
	

		//gradient descent rule: theta(j) = theta(j) - lambda * [partial derivative of cost function w.r.t. theta(j)]
		// set j th theta element. Update based on the calculation of partial gradient to minimize cost
		gsl_vector_float_set (&theta.vector, j, gsl_vector_float_get(&theta.vector, j)-(lambda*(partial_grad/m)) );
	}
}



/* This function actualy performs the training over the given data
	INPUT: theta matrix, y vector and input X matrix
	OUTPUT: Void
	
	Here a iteratively perform binary classification for each class i.e. for every class, it is taken as 1 while all other classes are taken as 0
*/
void train(gsl_matrix_float* thetamat, gsl_vector* y, gsl_matrix_float* X)
{
	int ite, iteration = 2;		// iterating to obtain theta vector as accurate as possible
	int class, i, m;

	gsl_vector* ytemp; 				// temporary vector y to perform training using binary classification
	m = y->size;
	ytemp = gsl_vector_alloc(m);  
   
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

		// adjust theta for this coverted binary classification
		for (ite = 1; ite <= iteration; ite++)
		{
			adjust_theta(theta, ytemp, X);  
		}

//check		printf("class %d done\n",class);
	}
   
    
	// as we are passing in a view, the theta matrix will automatically get updated, pass by reference

	// checking if pred is correct, ignore timne at present

	/* IO PORTION FOR PREDICT TESTING*/
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


	//  printf("now predicting");
	//predict(theta,y,X);
	//predict(thetamat,ytest,Xtest); 
}



/* Function to predict/classify the given input. Here it is a multivalue classification therefore prediction would be any value from 1 to 9
	INPUT: trained theta vector, y vector, input X matrix
	OUTPUT: Void
	
	Here 
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
