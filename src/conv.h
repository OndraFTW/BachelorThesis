/*
    Author: Ondřej Šlampa
    Email: o.slampa@gmail.com
    Description: Program that computes convolution of two signals.
*/

#include<stdbool.h>

//Method of computing convolution.
typedef enum {
	SINGLE_NAIVE,
	MULTI_NAIVE,
	FOURIER
} Method;

//Command line arguments.
typedef struct {
	bool valid;
	char* input_filename;
	char* kernel_filename;
	char* output_filename;
	bool help;
	Method method;
	unsigned int number_of_threads;
	bool input_columns;
	bool kernel_columns;
	bool output_columns;
} Arguments;

//Signal.
typedef struct {
    int number_of_dimensions;
    int* dimensions;
    int size;
    double* data;
} Signal;

//Signal with no information.
const Signal NULL_SIGNAL={
    .number_of_dimensions=0,
    .dimensions=NULL,
    .size=0,
    .data=NULL
};

