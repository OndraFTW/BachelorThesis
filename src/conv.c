/*
    Author: Ondřej Šlampa
    Email: o.slampa@gmail.com
    Description: Program that computes convolution of two signals.
*/

#include<ctype.h>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<stdbool.h>
#include<getopt.h>
#include<errno.h>
#include<omp.h>
#include<complex.h>
#include<fftw3.h>
#include<math.h>
#include<IL/il.h>
#include<IL/ilu.h>
#include"conv.h"

/*
    Parses command line arguments into Arguments structure.
    @param argc number of arguments
    @param argv arguments
    @return parsed arguments
*/
Arguments parse_command_line_options(int argc, char* argv[]){
	//default values
	Arguments args={
	    .valid=true,
	    .input_filename=NULL,
	    .kernel_filename=NULL,
	    .output_filename=NULL,
	    .help=false,
	    .method=SINGLE_NAIVE,
	    .number_of_threads=4,
	    .input_columns=false,
	    .kernel_columns=false,
	    .output_columns=false
	};
	
	//parse arguments 
	int c=0;
	while((c=getopt(argc, argv, "i:k:o:hm:t:abc")) != -1){
		switch(c){
			case 'i': args.input_filename=optarg; break;
			case 'k': args.kernel_filename=optarg; break;
			case 'o': args.output_filename=optarg; break;
			case 'h': args.help=true; break;
			case 'm': {
				if(strcmp(optarg, "single-naive")==0){
					args.method=SINGLE_NAIVE;
				}
				else if(strcmp(optarg, "multi-naive")==0){
					args.method=MULTI_NAIVE;
				}
				else if(strcmp(optarg, "fourier")==0){
					args.method=FOURIER;
				}
			}
			break;
			case 't': {
			    errno=0;
		        char *ep=NULL;
		        if((args.number_of_threads=strtol(optarg, &ep, 10))<=0 || errno!=0 || *ep!='\0'){
		            fprintf(stderr, "Invalid number of threads: '%s'.\n", optarg);
		            args.valid=false;
		        }
			}
			break;
			case 'a': args.input_columns=true; break;
			case 'b': args.kernel_columns=true; break;
			case 'c': args.output_columns=true; break;
		}
	}
	
	//validity control
	if(!args.help){
		if(args.input_filename==NULL){
			fprintf(stderr, "Missing input.\n");
			args.valid=false;
		}
		if(args.kernel_filename==NULL){
			fprintf(stderr, "Missing kernel.\n");
			args.valid=false;
		}
		if(args.output_filename==NULL){
			fprintf(stderr, "Missing output.\n");
			args.valid=false;
		}
	}
	
	return args;
}

/*
    Prints help message.
*/
void print_help(){
	printf(
		"Usage: c_conv [options]\n"
		"Options:\n"
		"  -i input      input signal\n"
		"  -k kernel     kernel\n"
		"  -o output     output signal\n"
		"  -a            input is stored in columns\n"
		"  -b            kernel is stored in columns\n"
		"  -c            store output in columns\n"
		"  -h            shows this help message\n"
		"  -m method     use method\n"
		"Methods:\n"
		"  single-naive  naive method using sigle thread\n"
		"  multi-naive   naive method using multiple threads\n"
		"  fourier       convolution theorem and fourier reansform\n"
	);
}

/*
    Switches rows and columns in signal.
    @param s signal
*/
void switch_rows_and_columns(Signal* s){
	//one-dimensional signals don't have rows and columns
	if(s->number_of_dimensions>=2){
		double* new_data=(double*)malloc(s->size*sizeof(double));
		int chunk_size=s->dimensions[s->number_of_dimensions-1]*s->dimensions[s->number_of_dimensions-2];
		int number_of_elements=1;
		for(int i=0; i<s->number_of_dimensions; i++){
			number_of_elements*=s->dimensions[i];
		}
		
		for(int chunk_i=0; chunk_i<number_of_elements/chunk_size; chunk_i++){
			for(int y=0; y<s->dimensions[s->number_of_dimensions-2]; y++){
				for(int x=0; x<s->dimensions[s->number_of_dimensions-1]; x++){
					new_data[chunk_i*chunk_size+y*s->dimensions[s->number_of_dimensions-1]+x]=
					    s->data[chunk_i*chunk_size+x*s->dimensions[s->number_of_dimensions-2]+y];
				}
			}
		}
		
		free(s->data);
		s->data=new_data;
		int tmp=s->dimensions[s->number_of_dimensions-2];
		s->dimensions[s->number_of_dimensions-2]=s->dimensions[s->number_of_dimensions-1];
		s->dimensions[s->number_of_dimensions-1]=tmp;
	}
}

/*
    Loads signal from text file.
    @param filename path to file
    @return loaded signal of NULL_SIGNAL if error occured
*/
Signal load_text(char* filename){
    Signal r;
    int allocated_dimensions=2;
    r.number_of_dimensions=1;
    r.dimensions=(int*)malloc(allocated_dimensions*sizeof(int));
    int scanned_dimension=0;
    int scanned_return=0;
    
    //open file
    FILE* file=fopen(filename, "r");
    if(file==NULL){
    	fprintf(stderr, "File '%s': can't be opened.\n", filename);
    	free(r.dimensions);
    	return NULL_SIGNAL;
    }
    
    //load size of first dimension
    scanned_return=fscanf(file, "%d", &scanned_dimension);
    r.dimensions[0]=scanned_dimension;
    r.size=scanned_dimension;
    
    if(scanned_return<=0){
    	fprintf(stderr, "File '%s': can't find size of first dimension.\n", filename);
    	free(r.dimensions);
    	fclose(file);
    	return NULL_SIGNAL;
    }
    
    if(scanned_dimension<=0){
    	fprintf(stderr, "File '%s': dimension size can't be 0 or lower.\n", filename);
    	free(r.dimensions);
    	fclose(file);
    	return NULL_SIGNAL;
    }
    
    //load sizes of other dimensions
    for(int i=1; fscanf(file, ",%d", &scanned_dimension)==1; i++){
    	if(i>=allocated_dimensions){
    		allocated_dimensions*=2;
    		r.dimensions=(int*)realloc(r.dimensions, allocated_dimensions*sizeof(int));
    	}
    	r.size*=scanned_dimension;
    	r.dimensions[i]=scanned_dimension;
    	r.number_of_dimensions++;
    	
    	if(scanned_dimension<=0){
    		fprintf(stderr, "File '%s': dimension size can't be 0 or lower.\n", filename);
    		free(r.dimensions);
    		fclose(file);
    		return NULL_SIGNAL;
    	}
    }
    
    //compute total number of elements
    int number_of_elements=1;
    for(int i=0; i<r.number_of_dimensions; i++){
    	number_of_elements*=r.dimensions[i];
    }
    
    //allocate array for data
    r.data=(double*)malloc(number_of_elements*sizeof(double));
    
    int next_character=0;
    double scanned_value=0.0;

    //load elements
    for(int i=0; i<number_of_elements; i++){
        //skip ',' and white space characters
        do{
        	next_character=getc(file);
        }
        while(next_character==',' || isspace(next_character));
        ungetc(next_character, file);
        
        scanned_return=fscanf(file, "%lf", &scanned_value);
        r.data[i]=scanned_value;
        
        if(scanned_return<=0){
    		fprintf(stderr, "File '%s': missing element.\n", filename);
    		free(r.dimensions);
    		free(r.data);
    		fclose(file);
    		return NULL_SIGNAL;
    	}
    }
    
    fclose(file);
    return r;
}

/*
    Loads signal from image file.
    @param filename path to file
    @return loaded signal of NULL_SIGNAL if error occured
*/
Signal load_image(char* filename){
    Signal r;
    //initialize image in DevIL
    ILuint imagename;
    ilGenImages(1, &imagename);
    ilBindImage(imagename);
    //load file
    if(ilLoadImage(filename)==IL_FALSE){
    	fprintf(stderr, "File '%s': non-existent file or uknown file extension.\n", filename);
    	return NULL_SIGNAL;
    }
    if(ilGetError()!=IL_NO_ERROR){
    	switch(ilGetError()){
    		case IL_COULD_NOT_OPEN_FILE: printf("File '%s': can't be opened.\n", filename); break;
    		case IL_ILLEGAL_OPERATION: printf("File '%s': no image.\n", filename); break;
    		case IL_INVALID_EXTENSION: printf("File '%s': invalid file extension.\n", filename); break;
    		case IL_INVALID_PARAM: printf("File '%s': invalid filename.\n", filename); break;
    		default: printf("File '%s': uknown DevIL error.\n", filename); break;
    	}
    	return NULL_SIGNAL;
    }
    //load dimensions
    r.number_of_dimensions=2;
    r.dimensions=(int*)malloc(2*sizeof(int));
    r.dimensions[1]=ilGetInteger(IL_IMAGE_WIDTH);
    r.dimensions[0]=ilGetInteger(IL_IMAGE_HEIGHT);
    r.size=r.dimensions[1]*r.dimensions[0];
    //load elements
    r.data=(double*)malloc(r.dimensions[0]*r.dimensions[1]*sizeof(double));
    ilCopyPixels(0,0,0,r.dimensions[1],r.dimensions[0],1,IL_LUMINANCE,IL_DOUBLE,r.data);
    if(ilGetError()!=IL_NO_ERROR){
    	switch(ilGetError()){
    		case IL_ILLEGAL_OPERATION: printf("File '%s': illegal operation.\n", filename); break;
    		case IL_INVALID_PARAM: printf("File '%s': invalid filename.\n", filename); break;
    		default: printf("File '%s': uknown DevIL error.\n", filename); break;
    	}
    	return NULL_SIGNAL;
    }
    return r;
}

/*
    Gets last n characters from string.
    @param string source string
    @param n number of characters
    @returns last n characters od string
*/
char* last_n_chars(char* string, int n){
	int len=strlen(string);
	return &string[len-n];
}

/*
    Loads signal from file.
    @param filename path to file
    @return loaded signal of NULL_SIGNAL if error occured
*/
Signal load_signal(char* filename, bool switch_rc){
	Signal tmp;
	char* last_4=last_n_chars(filename, 4);
	
	if(strcmp(last_4, ".csv")==0 || strcmp(last_4, ".txt")==0){
		tmp=load_text(filename);
		if(switch_rc){
			switch_rows_and_columns(&tmp);
		}
		return tmp;
	}
	else{
		tmp=load_image(filename);
		if(switch_rc){
			switch_rows_and_columns(&tmp);
		}
		return tmp;
	}
}

/*
    Recursively saves n-dimensional signal into text file.
    @param file destination file
    @param data signal data
    @param dimensions dimensions of signal
    @param depth number of dimensions
*/
void save_dimension(FILE* file, double* data, int* dimensions, int depth){
	//save one-dimensional signal
	if(depth==1){
		fprintf(file, "%lf", data[0]);
		for(int i=1; i<dimensions[0]; i++){
			fprintf(file, ",%lf", data[i]);
		}
	}
	//save n-dimensional signal as list of (n-1)-dimensional signals
	else{
		int  number_of_elements=1;
		for(int i=1; i<depth; i++){
			number_of_elements*=dimensions[i];
		}
		for(int i=0; i<dimensions[0]; i++){
			save_dimension(file, &data[number_of_elements*i], &dimensions[1], depth-1);
		}
	}
	fprintf(file, "\n");
}

/*
    Saves signal into text file.
    @param signal source signal
    @param filename path to file
*/
void save_text(Signal* signal, char* filename){
	//open file
	FILE* file=fopen(filename, "w");
	if(file==NULL){
		fprintf(stderr, "File '%s': can't be openend.\n", filename);
		return;
	}
	//save dimensions
	fprintf(file, "%d", signal->dimensions[0]);
	for(int i=1; i<signal->number_of_dimensions; i++){
		fprintf(file, ",%d", signal->dimensions[i]);
	}
	fprintf(file, "\n");
	//save elements
	save_dimension(file, signal->data, signal->dimensions, signal->number_of_dimensions);
	fclose(file);
}

/*
    Saves signal into image file.
    @param signal source signal
    @param filename path to file
*/
void save_image(Signal* image, char* filename){
	//initialize DevIL image
	ILuint imagename;
	ilGenImages(1, &imagename);
    ilBindImage(imagename);
    //load signal into DevIL image
	ilTexImage(image->dimensions[1],image->dimensions[0],1,1,IL_LUMINANCE,IL_DOUBLE,image->data);
	if(ilGetError()!=IL_NO_ERROR){
    	switch(ilGetError()){
    		case IL_ILLEGAL_OPERATION: printf("File '%s': illegal operation.\n", filename); break;
    		case IL_INVALID_PARAM: printf("File '%s': invalid parameter.\n", filename); break;
    		case IL_OUT_OF_MEMORY: printf("File '%s': out of memory.\n", filename); break;
    		default: printf("File '%s': ilTexImage(): uknown DevIL error.\n", filename); break;
    	}
    	return;
    }
    //flip image
	iluFlipImage();
	if(ilGetError()!=IL_NO_ERROR){
    	switch(ilGetError()){
    		case IL_ILLEGAL_OPERATION: printf("File '%s': illegal operation.\n", filename); break;
			case IL_OUT_OF_MEMORY: printf("File '%s': out of memory.\n", filename); break;
    		default: printf("File '%s': iluFlipImage(): uknown DevIL error.\n", filename); break;
    	}
    	return;
    }
    //save image
	ilSaveImage(filename);
	if(ilGetError()!=IL_NO_ERROR){
    	switch(ilGetError()){
    		case IL_ILLEGAL_OPERATION: printf("File '%s': illegal operation.\n", filename); break;
    		case IL_INVALID_PARAM: printf("File '%s': invalid filename.\n", filename); break;
    		case IL_COULD_NOT_OPEN_FILE: printf("File '%s': can't open the file.\n", filename); break;
			case IL_INVALID_EXTENSION: printf("File '%s': invalid extension.\n", filename); break;    		
    		default: printf("File '%s': ilSaveImage(): uknown DevIL error.\n", filename); break;
    	}
    }
}

/*
    Saves signal into file.
    @param signal source signal
    @param filename path to file
*/
void save_signal(Signal* signal, char* filename, bool switch_rc){
	char* last_4=last_n_chars(filename, 4);
	if(switch_rc){
		switch_rows_and_columns(signal);
	}
	if(strcmp(last_4, ".csv")==0 || strcmp(last_4, ".txt")==0){
		save_text(signal, filename);
	}
	else{
		save_image(signal, filename);
	}
}

/*
    Equalizes number of dimensions of two signal. Signal with lower number of
    dimensions is extended. Size of added dimensions is 1.
    @param s1 first signal
    @param s2 second signal
*/
void equalize_number_of_dimensions(Signal* s1, Signal* s2){
	//extend second signal
	if(s1->number_of_dimensions>s2->number_of_dimensions){
		int difference=s1->number_of_dimensions-s2->number_of_dimensions;
		s2->dimensions=(int*)realloc(s2->dimensions, s1->number_of_dimensions*sizeof(int));
		for(int i=s2->number_of_dimensions-1; i>=0; i--){
			s2->dimensions[i+difference]=s2->dimensions[i];
		}
		for(int i=0; i<difference; i++){
			s2->dimensions[i]=1;
		}
		s2->number_of_dimensions=s1->number_of_dimensions;
	}
	//extend first signal
	else if(s2->number_of_dimensions>s1->number_of_dimensions){
		int difference=s2->number_of_dimensions-s1->number_of_dimensions;
		s1->dimensions=(int*)realloc(s1->dimensions, s2->number_of_dimensions*sizeof(int));
		for(int i=s1->number_of_dimensions-1; i>=0; i--){
			s1->dimensions[i+difference]=s1->dimensions[i];
		}
		for(int i=0; i<difference; i++){
			s1->dimensions[i]=1;
		}
		s1->number_of_dimensions=s2->number_of_dimensions;
	}
}

/*
    Recursively transforms n-dimensional signal to one-dimensional signal.
    @param old_data old signal
    @param new_data new signal
    @param old_dimensions dimensions of old signal
    @param new_dimensions dimensions of new signal
    @param dept number of dimensions
    @param last current ND signal is last signal in (N+1)D signal
    @return number of elements of new signal
*/
int transform(double* old_data, double* new_data, int* old_dimensions, int* new_dimensions, int depth, bool last){
	//one-dimensional signal is copied and extended with zeros.
	if(depth==1){
		for(int i=0; i<old_dimensions[0]; i++){
			new_data[i]=old_data[i];
		}
		//if this isn't last subsignal add zeros
		if(last){
			return old_dimensions[0];
		}
		else{
			for(int i=old_dimensions[0]; i<new_dimensions[0]; i++){
				new_data[i]=0.0;
			}
			return new_dimensions[0];
		}
	}
	//n-dimensional signal is transforned as list of (n-1)-dimensional signals.
	else{
	    //size of new signal
		int size=0;
		//compute number of elements
		int number_of_new_elements=1;
		int number_of_old_elements=1;
		for(int i=1; i<depth; i++){
			number_of_new_elements*=new_dimensions[i];
			number_of_old_elements*=old_dimensions[i];
		}
		//this is last signal
		if(last){
			//transform subsignals except the last one
			for(int i=0; i<old_dimensions[0]-1; i++){
				size+=transform(
					&old_data[i*number_of_old_elements],
					&new_data[i*number_of_new_elements],
					&old_dimensions[1], &new_dimensions[1], depth-1, false
				);
			}
			//transform last subsignal
			size+=transform(
				&old_data[(old_dimensions[0]-1)*number_of_old_elements],
				&new_data[(old_dimensions[0]-1)*number_of_new_elements],
				&old_dimensions[1], &new_dimensions[1], depth-1, true
			);
		}
		//this isn't last signal
		else{
			//transform subsignals
			for(int i=0; i<old_dimensions[0]; i++){
				size+=transform(
					&old_data[i*number_of_old_elements],
					&new_data[i*number_of_new_elements],
					&old_dimensions[1], &new_dimensions[1], depth-1, false
				);
			}
			//add zeros
			for(int i=old_dimensions[0]; i<new_dimensions[0]; i++){
				for(int j=0; j<number_of_new_elements; j++){
					new_data[i*number_of_new_elements+j]=0.0;
				}
			}
			size+=(new_dimensions[0]-old_dimensions[0])*number_of_new_elements;
		}
		return size;
	}
}

/*
    Transforms two n-dimensional signals to one dimensional signals.
    @param s1 first signal
    @param s2 second signal
    @param cut_zeros cut trailing zeros
*/
void transform_signals_to_1D(Signal* s1, Signal* s2, bool cut_zeros){
	int number_of_dimensions=s1->number_of_dimensions;
	//compute length of new signals with trailing zeros
	int number_of_elements=1;
	int* dimensions=(int*)malloc(number_of_dimensions*sizeof(int));
	for(int i=0; i<number_of_dimensions; i++){
		dimensions[i]=s1->dimensions[i]+s2->dimensions[i]-1;
		number_of_elements*=dimensions[i];
	}
	//allocate memory
	double* new_data1=(double*)malloc(number_of_elements*sizeof(double));
	double* new_data2=(double*)malloc(number_of_elements*sizeof(double));
    
    //transformation
	#pragma omp parallel sections
	{
		s1->size=transform(s1->data, new_data1, s1->dimensions, dimensions, number_of_dimensions, cut_zeros);
		#pragma omp section
		s2->size=transform(s2->data, new_data2, s2->dimensions, dimensions, number_of_dimensions, cut_zeros);
	}
	s1->dimensions=(int*)malloc(sizeof(int));
	s2->dimensions=(int*)malloc(sizeof(int));
	s1->dimensions[0]=s1->size;
	s2->dimensions[0]=s2->size;
	s1->data=new_data1;
	s2->data=new_data2;
	s1->number_of_dimensions=1;
	s2->number_of_dimensions=1;
	free(dimensions);
}

/*
    Transforms one-dimensional signal to n-dimensional.
    @param s signal
    @param number_of_dimensions number of dimensions
    @param dimensions dimensions of new signal
*/
void transform_signal_to_ND(Signal* s, int number_of_dimensions, int* dimensions){
	s->number_of_dimensions=number_of_dimensions;
	s->dimensions=dimensions;
}

/*
    Convolves two signals.
    @param input input signal
    @param kernel kernel signal
    @param use_multiple_threads should function use multiple threads
    @return result of convolution of input and kernel
*/
Signal convolve(Signal input, Signal kernel, bool use_multiple_threads, unsigned int number_of_threads){
	Signal output;
	output.number_of_dimensions=1;
	output.size=input.size+kernel.size-1;
	output.dimensions=(int*)malloc(sizeof(int));
	output.dimensions[0]=output.size;
	output.data=(double*)malloc(output.size*sizeof(double));
	
	//compute convolution
	#pragma omp parallel for if(use_multiple_threads) num_threads(number_of_threads)
	for(int n=0; n<output.size; n++){
		double sum=0.0;
		for(int i=0; i<kernel.size; i++){
			int a=n-i;
			if(n<i){
				break;
			}
			else if(a>=input.size){
				i=n-input.size;
			}
			else{
				double e=input.data[a];
				double f=kernel.data[i];
				sum+=e*f;
			}
		}
		#pragma omp critical
		output.data[n]=sum;
	}
	
	return output;
}

/*
    Convolves two signals using FFTW library.
    @param input input signal
    @param kernel kernel signal
    @return result of convolution of input and kernel
*/
Signal fftw_convolve(Signal input, Signal kernel){
	fftw_complex *cinput, *ckernel;
    fftw_plan p1, p2;
    
    //transform signals to FFTW data types
    cinput=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*input.size);
    ckernel=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*kernel.size);
    
    #pragma omp parallel sections
    {
		for(int i=0; i<input.size; i++){
			cinput[i]=(fftw_complex)input.data[i];
		}
		#pragma omp section
		for(int i=0; i<kernel.size; i++){
			ckernel[i]=(fftw_complex)kernel.data[i];
		}
    }
    
    //transform signals with FFT
    p1=fftw_plan_dft_1d(input.size, cinput, cinput, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p1);
    fftw_destroy_plan(p1);
    
    p2=fftw_plan_dft_1d(kernel.size, ckernel, ckernel, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p2);
    fftw_destroy_plan(p2);
    
    //multiply signals
    for(int i=0; i<input.size; i++){
    	cinput[i]*=ckernel[i];
    }
    
    //transform result with IFFT
    p1=fftw_plan_dft_1d(input.size, cinput, cinput, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p1);
    fftw_destroy_plan(p1);
    
    fftw_free(ckernel);
    
    //transform result to C types and Signal struct
    Signal result;
    result.number_of_dimensions=1;
    result.size=input.size;
    result.dimensions=(int*)malloc(sizeof(int));
    result.dimensions[0]=result.size;
    result.data=(double*)malloc(result.size*sizeof(double));
    for(int i=0; i<result.size; i++){
    	result.data[i]=((double)cinput[i])/result.size;
    }
    
    fftw_free(cinput);
    return result;
}

/*
    Main function.
    @param argc number of arguments
    @param argv arguments
    @return EXIT_SUCCESS or EXIT_FAILURE
*/
int main(int argc, char* argv[]){
    //initialize DevIL
    ilInit();
    ilEnable(IL_FILE_OVERWRITE);
    ilEnable(IL_ORIGIN_SET);
    ilOriginFunc(IL_ORIGIN_UPPER_LEFT);
    
    //parse command line arguments
    Arguments args=parse_command_line_options(argc, argv);
    
    if(!args.valid){
    	return EXIT_FAILURE;
    }
    
    if(args.help){
    	print_help();
    	return EXIT_SUCCESS;
    }
    
    //load signals
    Signal kernel;
    Signal input;
    #pragma omp parallel sections
    {
    	kernel=load_signal(args.kernel_filename, args.kernel_columns);
    	#pragma omp section
    	input=load_signal(args.input_filename, args.input_columns);
    }
    
    if(input.data==NULL && kernel.data==NULL){
    	return EXIT_FAILURE;
    }
    else if(input.data==NULL){
    	free(kernel.dimensions);
    	free(kernel.data);
    	return EXIT_FAILURE;
    }
    else if(kernel.data==NULL){
    	free(input.dimensions);
    	free(input.data);
    	return EXIT_FAILURE;
    }
    
    //equalize number of dimensions
    if(input.number_of_dimensions!=kernel.number_of_dimensions){
    	equalize_number_of_dimensions(&input, &kernel);
    }
    
    //transform signals to 1D
    Signal input1D=input;
    Signal kernel1D=kernel;
    transform_signals_to_1D(&input1D, &kernel1D, args.method!=FOURIER);
    Signal output1D;
    
    free(input.data);
    free(kernel.data);
    
    //compute dimensions of output signal
    int number_of_dimensions=input.number_of_dimensions;
	int* dimensions=(int*)malloc(number_of_dimensions*sizeof(int));
	for(int i=0; i<number_of_dimensions; i++){
		dimensions[i]=input.dimensions[i]+kernel.dimensions[i]-1;
	}
	
    free(input.dimensions);
    free(kernel.dimensions);
    
    //compute convolution
    switch(args.method){
    	case SINGLE_NAIVE: output1D=convolve(input1D, kernel1D, false, 1); break;
    	case MULTI_NAIVE: output1D=convolve(input1D, kernel1D, true, args.number_of_threads); break;
    	case FOURIER: output1D=fftw_convolve(input1D, kernel1D); break;
    }

    //transform output signal to ND
	Signal output=output1D;
	transform_signal_to_ND(&output, number_of_dimensions, dimensions);
    
    //save output signal
    save_signal(&output, args.output_filename, args.output_columns);
    
    free(input1D.data);
    free(input1D.dimensions);
    free(kernel1D.data);
    free(kernel1D.dimensions);
    free(output1D.dimensions);
    free(output.data);
    free(output.dimensions);
    
    return EXIT_SUCCESS;
}

