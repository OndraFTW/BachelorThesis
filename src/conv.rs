/*
    Author: Ondřej Šlampa
    Email: o.slampa@gmail.com
    Description: Program that computes convolution of two signals.
*/

extern crate std;
extern crate sync;
extern crate getopts;

use std::io::println;
use std::io::File;
use std::io::BufferedReader;
use std::io::IoResult;
use std::libc;
use std::c_str;
use std::slice;
use sync::Arc;

#[link(name="IL")]
#[link(name="ILU")]

//Extern functions definitions
extern{
    fn ilInit();
    fn ilGetError()->libc::c_uint;
    fn ilEnable(mode:libc::c_uint)->libc::c_uchar;
    fn ilOriginFunc(mode:libc::c_uint)->libc::c_uchar;
    fn ilGenImages(num:libc::size_t, images:*libc::c_uint);
    fn ilBindImage(image:libc::c_uint);
    fn ilLoadImage(filename:c_str::CString)->libc::c_uchar;
    fn ilSaveImage(filename:c_str::CString)->libc::c_uchar;
    fn ilGetInteger(mode:libc::c_uint)->libc::c_int;
    fn ilCopyPixels(
        x_off:libc::c_uint, y_off:libc::c_uint, z_off:libc::c_uint,
        width:libc::c_uint, height:libc::c_uint, depth:libc::c_uint,
        format:libc::c_uint, type_:libc::c_uint, data:*libc::c_void
    );
    fn ilTexImage(
         width:libc::c_uint, height:libc::c_uint, depth:libc::c_uint,
         number_of_Sendernels:libc::c_uchar, format:libc::c_uint, type_:libc::c_uint,data:*libc::c_void
    );
    fn iluFlipImage()->libc::c_uchar;
}

//Different algorithms that can be used to compute convolution.
enum Method{
    SingleNaive,
    MultiNaive,
    OverlapSave
}

//Different actions that can be performed by worker thread.
enum Action{
    Compute,
    Terminate
}

//Program arguments.
struct Arguments{
    input_filename:~str,
    kernel_filename:~str,
    output_filename:~str,
    help:bool,
    method:Method,
    number_of_workers:uint,
    input_columns:bool,
    kernel_columns:bool,
    output_columns:bool
}

//Signal.
struct Signal{
    dimensions:~[uint],
    data:~[f64]
}

/*
    Parses command line arguments.
    returns: parsed arguments
*/
fn parse_command_line_arguments()->~Arguments{
    let args=std::os::args();
    
    let opts=~[
        getopts::optopt("i", "input", "input signal", "FILE"),
        getopts::optopt("k", "kernel", "kernel", "FILE"),
        getopts::optopt("o", "output", "output signal", "FILE"),
        getopts::optopt("m", "method", "use method", "METHOD"),
        getopts::optopt("w", "workers", "use NUMBER worker threads", "NUMBER"),
        getopts::optflag("h", "help", "shows this help message"),
        getopts::optflag("a", "input-columns", "input is stored columns"),
        getopts::optflag("b", "kernel-columns", "kernel is stored columns"),
        getopts::optflag("c", "output-columns", "store output in columns")
    ];
    
    let matches=match getopts::getopts(args.tail(), opts){
        Ok(m)=>{
            m
        }
        Err(f)=>{
            fail!(f.to_err_msg())
        }
    };

    return ~Arguments{
        input_filename:match matches.opt_str("i"){
        	None=>~"input",
        	Some(a)=>a
        },
        kernel_filename:match matches.opt_str("k"){
        	None=>~"kernel",
        	Some(a)=>a
        },
        output_filename:match matches.opt_str("o"){
        	None=>~"output",
        	Some(a)=>a
        },
        help:matches.opt_present("h"),
        method:if matches.opt_str("m").is_some(){
            let method:~str=matches.opt_str("m").unwrap();
            if method.eq(&~"single-naive"){
                SingleNaive
            }
            else if method.eq(&~"multi-naive"){
                MultiNaive
            }
            else if method.eq(&~"overlap-save"){
                OverlapSave
            }
            else{
                SingleNaive
            }
        }
        else{
            SingleNaive
        },
        number_of_workers:match matches.opt_str("w"){
            Some(s)=>match from_str(s){
                Some(0u)=>4u,
                Some(n)=>n,
                None=>4u
            },
            None=>4u
        },
        input_columns:matches.opt_present("a"),
        kernel_columns:matches.opt_present("b"),
        output_columns:matches.opt_present("c")
    };
}

/*
    Prints help message.
*/
fn print_help(){
    println(
		"Usage: rust_conv [options]\n"+
		"Options:\n"+
		"  -i input      input signal\n"+
		"  -k kernel     kernel\n"+
		"  -o output     output signal\n"+
		"  -a            input is stored columns\n"+
		"  -b            kernel is stored columns\n"+
		"  -c            store output in columns\n"+
		"  -h            shows this help message\n"+
		"  -m method     use method\n"+
		"  -w number     use number worker threads\n"+
		"Methods:\n"+
		"  single-naive  naive method using sigle thread\n"+
		"  multi-naive   naive method using multiple threads\n"+
		"  overlap-save  overlap-save using multiple threads"
    );
}

/*
    Initializes DevIL.
*/
fn init_devil(){
    unsafe{
        ilInit();
        ilEnable(0x0620);
        ilEnable(0x0600);
        ilOriginFunc(0x0602);
    }
}

/*
    Loads signal from text file.
    filename: path to the file
    columns: signal is stored in columns
    returns: loaded signal or error message
*/
fn load_text(filename:&str, columns:bool)->Result<~Signal, ~str>{
    //open file
    let path=~Path::new(filename);
    if !path.exists(){
        return Err(format!("File '{}': doesn't exist.", filename));
    }
    let file=File::open(path);
    let mut reader=BufferedReader::new(file);
    //load dimensions
    let line=reader.read_line();
    let mut dimensions:~[uint]=~[];
    match line{
        Err(_)=>{
            return Err(format!("File '{}': missing first line.", filename));
        }
        Ok(line)=>{
            for string in line.split_str(","){
                let option:Option<uint> =from_str(string.trim());
                match option{
                    Some(val) if val==0=>return Err(format!("File '{}': dimension size can't be zero.", filename)),
                    Some(val)=>dimensions.push(val),
                    None=>return Err(format!("File '{}': first line format error.", filename))
                }
            }
        }
    }
    //number of elements
    let size=dimensions.iter().fold(1u, |a:uint,b:&uint|a*(*b));
    let mut data:~[f64]=slice::from_elem(size, 0.0f64);
    let mut n=0u;
    
    //load elements of signal
    for maybe_line in reader.lines(){
        match maybe_line{
            Ok(line)=>{
                if line.trim()!=""{
                    for string in line.split_str(","){
                        let option:Option<f64> =from_str(string.trim());
                        match option{
                            Some(num)=>{
                                data[n]=num;
                                n+=1;
                            }
                            None=>{
                                return Err(format!("File '{}': format error.", filename))
                            }
                        }
                    }
                }
            }
            Err(_)=>{
            
            }
        }
    }
    
    //check number of elements
    if n!=size{
        return Err(format!("File '{}': wrong number of elements.", filename))
    }

    if columns{
        return Ok(switch_rows_and_columns(~Signal{
            data:data,
            dimensions:dimensions
        }));
    }
    else{
        return Ok(~Signal{
            dimensions:dimensions,
            data:data
        });
    }
}

/*
    Loads signal from image file.
    filename: path to the file
    columns: signal is stored in columns
    returns: loaded signal or error message
*/
fn load_image(filename:&str, columns:bool)->Result<~Signal, ~str>{
    //open file
    let c_filename=filename.to_c_str();
    let imagename:~[libc::c_uint]=~[0 as libc::c_uint];
    unsafe{
        ilGenImages(1 as libc::size_t, imagename.as_ptr() as *libc::c_uint);
        ilBindImage(imagename[0]);
        if ilLoadImage(c_filename)==0{
            return Err(format!("File '{}': non-existent file or uknown file extension.", filename));
        }
        if ilGetError()!=0{
        	match ilGetError(){
        		0x050A=>return Err(format!("File '{}': can't be opened.", filename)),
        		0x0506=>return Err(format!("File '{}': no image.", filename)),
        		0x050B=>return Err(format!("File '{}': invalid file extension.", filename)),
        		0x0509=>return Err(format!("File '{}': invalid filename.", filename)),
        		_=>return Err(format!("File '{}': uknown DevIL error.", filename))
        	}
        }
    };
    //load dimensions
    let height=unsafe{
        ilGetInteger(0x0DE5)
    } as uint;
    let width=unsafe{
        ilGetInteger(0x0DE4)
    } as uint;

    let data:~[libc::c_double]=slice::from_elem(height*width, 0.0 as libc::c_double);
    
    //load elements
    unsafe{
        ilCopyPixels(
            0,0,0,width as libc::c_uint,height as libc::c_uint,1,
            0x1909,0x140A,data.as_ptr() as *libc::c_void
        );
        if ilGetError()!=0{
        	match ilGetError(){
        		0x0506=>return Err(format!("File '{}': illegal operation.\n", filename)),
        		0x0509=>return Err(format!("File '{}': invalid filename.\n", filename)),
        		_=>return Err(format!("File '{}': uknown DevIL error.\n", filename))
        	}
        }
    };
    
    //convert elemets to Rust data types
    let data_f64:~[f64]=slice::from_fn(height*width, |i:uint|{
        data[i] as f64
    });

	if columns{
		return Ok(switch_rows_and_columns(~Signal{
    	    data:data_f64,
    	    dimensions:~[height, width]
    	}));
	}
	else{
    	return Ok(~Signal{
    	    data:data_f64,
    	    dimensions:~[height, width]
    	});
    }
}

/*
    Switches rows and columns of signal.
    s: input signal
    returns: output signal
*/
fn switch_rows_and_columns(s:~Signal)->~Signal{
    if s.dimensions.len()>=2{
    	let data_len=s.data.len();
    	let mut new_dimensions=s.dimensions.clone();
    	new_dimensions.reverse();
    	let tmp=new_dimensions[0];
    	new_dimensions[0]=new_dimensions[1];
    	new_dimensions[1]=tmp;
    	let mut new_data=slice::from_elem(data_len, 0.0f64);
    	let chunk_size=new_dimensions[0]*new_dimensions[1];
    	for chunk_i in range(0u, data_len/chunk_size){
    		for i in range(0u, chunk_size){
    			let x=i%new_dimensions[0];
            	let y=i/new_dimensions[0];
            	new_data[chunk_i*chunk_size+y*new_dimensions[0]+x]=s.data[chunk_i*chunk_size+x*new_dimensions[0]+y]
    		}
    	}
    	new_dimensions.reverse();
    	return ~Signal{
    		dimensions:new_dimensions,
    		data:new_data
    	};
    }
    else{
        return s;
    }
}

/*
    Loads signal from file.
    filename: path to the file
    columns: signal is stored in columns
    returns: loaded signal or error message
*/
fn load_signal(filename:&str, columns:bool)->Result<~Signal, ~str>{
    if filename.ends_with(".csv") || filename.ends_with(".txt"){
        return load_text(filename, columns);
    }
    else{
        return load_image(filename, columns);
    }
}

/*
    Saves signal into image file.
    s: signal
    filename: path to the file
    returns: succes message or error message
*/
fn save_image(s:&Signal, filename:&str)->Result<~str, ~str>{
    let imagename:~[libc::c_uint]=~[0 as libc::c_uint];
    //convert signal elements to C types
    let c_data:~[libc::c_double]=slice::from_fn(s.dimensions[0]*s.dimensions[1], |i:uint|{
        s.data[i] as libc::c_double
    });

    unsafe{
        //load image into DevIL
        ilGenImages(1 as libc::size_t, imagename.as_ptr() as *libc::c_uint);
        ilBindImage(imagename[0]);
        ilTexImage(
            s.dimensions[1] as libc::c_uint, s.dimensions[0] as libc::c_uint, 1 as libc::c_uint,
            1 as libc::c_uchar, 0x1909, 0x140A, c_data.as_ptr() as *libc::c_void
        );
        if ilGetError()!=0{
        	match ilGetError(){
        		0x0506=>return Err(format!("File '{}': illegal operation.", filename)),
        		0x0509=>return Err(format!("File '{}': invalid parameter.", filename)),
        		0x0502=>return Err(format!("File '{}': out of memory.", filename)),
        		_=>return Err(format!("File '{}': ilTexImage(): uknown DevIL error.", filename))
        	}
        }
        //flip image
        iluFlipImage();
        if ilGetError()!=0{
        	match ilGetError(){
        		0x0506=>return Err(format!("File '{}': illegal operation.", filename)),
			    0x0502=>return Err(format!("File '{}': out of memory.", filename)),
        		_=>return Err(format!("File '{}': iluFlipImage(): uknown DevIL error.", filename))
        	}
        }
        //save image
        ilSaveImage(filename.to_c_str());
        if ilGetError()!=0{
        	match ilGetError(){
        		0x0506=>return Err(format!("File '{}': illegal operation.", filename)),
        		0x0509=>return Err(format!("File '{}': invalid filename.", filename)),
        		0x050A=>return Err(format!("File '{}': can't open the file.", filename)),
			    0x050B=>return Err(format!("File '{}': invalid extension.", filename)),    		
        		_=>return Err(format!("File '{}': ilSaveImage(): uknown DevIL error.", filename))
        	}
        }
    };
    
    return Ok(~"");
}

/*
    Saves signal into file.
    s: signal
    filename: path to the file
    columns: save image in columns
    returns: succes message or error message
*/
fn save_signal(s:~Signal, filename:&str, columns:bool)->Result<~str, ~str>{
    let mut r=s;
    if columns{
    	r=switch_rows_and_columns(r);
    }
    if filename.ends_with(".csv") || filename.ends_with(".txt"){
        return save_text(r, filename);
    }
    else{
        return save_image(r, filename);
    }

}

/*
    Recursively saves n-dimensional signal into text file.
    writer: destination file writer
    data: signal data
    dimensions: dimensions of signal
*/
fn save_dimension(writer:&mut File, data:&[f64], dimensions:&[uint])->IoResult<()>{
    //save one-dimensional signal
    let mut write_result:IoResult<()>;
    if dimensions.len()==1{
        let mut it=data.iter();
        match it.next(){
            Some(option)=>{
                write_result=writer.write_str(option.to_str());
                if write_result.is_err(){
                    return write_result;
                }
                for n in it{
                    write_result=writer.write_str(",");
                    if write_result.is_err(){
                        return write_result;
                    }
                    write_result=writer.write_str(n.to_str());
                    if write_result.is_err(){
                        return write_result;
                    }
                }
                write_result=writer.write_str("\n");
                if write_result.is_err(){
                    return write_result;
                }
            }
            None=>{
                fail!("Internal error: saving dimension.");
            }
        }
    }
    //save n-dimensional signal as list of (n-1)-dimensional signals
    else{
        let mut it=data.chunks(dimensions[0]);
        for chunk in it{
            write_result=save_dimension(writer, chunk, dimensions.slice_from(1));
            if write_result.is_err(){
                return write_result;
            }
        }
        write_result=writer.write_str("\n");
        if write_result.is_err(){
            println("Error");
        }
    }
    return Ok(());
}

/*
    Saves signal into text file.
    s: signal
    filename: path to the file
    returns: succes message or error message
*/
fn save_text(s:&Signal, filename:&str)->Result<~str, ~str>{
    //open file
    let maybe_writer=File::create(&Path::new(filename));
    let mut write_result:IoResult<()>;
    match maybe_writer{
        Err(_)=>{
            return Err(format!("File '{}': writer can't be created.", filename));
        }
        Ok(writer)=>{
            let mut writer=~writer;
            let mut it=s.dimensions.iter();
            match it.next(){
                None=>{
                    return Err(format!("File '{}': no dimensions.", filename));
                }
                Some(option)=>{
                    //save first dimension
                    write_result=writer.write_str(option.to_str());
                    if write_result.is_err(){
                        return Err(format!("File '{}': IoError.", filename));
                    }
                    //save other dimensions
                    for dim in it{
                        write_result=writer.write_str(",");
                        if write_result.is_err(){
                            return Err(format!("File '{}': IoError.", filename));
                        }
                        write_result=writer.write_str(dim.to_str());
                        if write_result.is_err(){
                            return Err(format!("File '{}': IoError.", filename));
                        }
                    }
                    write_result=writer.write_str("\n");
                    if write_result.is_err(){
                        return Err(format!("File '{}': IoError.", filename));
                    }
                    //save elements
                    write_result=save_dimension(writer, s.data, s.dimensions);
                    if write_result.is_err(){
                        return Err(format!("File '{}': IoError.", filename));
                    }
                    return Ok(~"");
                }
            }
        }
    }
}

/*
    Equalizes number of dimensions of two signal. Signal with lower number of
    dimensions is extended. Size of added dimensions is 1.
    s1: first signal
    s2: second signal
    returns: equalized signals
*/
fn equalize_number_of_dimensions(s1:~Signal, s2:~Signal)->(~Signal, ~Signal){
    let difference=s1.dimensions.len()-s2.dimensions.len();
    //extend second signal
    if difference>0{
        return (s1, ~Signal{
            dimensions:slice::append(slice::from_elem(difference, 1u), s2.dimensions),
            data:s2.data
        });
    }
    //extend first signal
    else if difference<0{
        return (~Signal{
            dimensions:slice::append(slice::from_elem(-difference, 1u), s1.dimensions),
            data:s1.data
        },
        s2
        );
    }
    else{
        return (s1, s2);
    }
}

/*
    Recursively transforms n-dimensional signal to one-dimensional signal.
    old_data: old signal
    old_offset: index of first element of old signal
    new_data: new signal
    new_offset: index of first element of new signal
    old_dimensions: dimensions of old signal
    new_dimensions: dimensions of new signal
    last: current ND signal is last signal in (N+1)D signal
    returns: number of elements of new signal
*/
fn transform(old_data:&[f64], old_offset:uint, new_data:&mut[f64], new_offset:uint,
                old_dimensions:&[uint], new_dimensions:&[uint], last:bool)->uint{
    //one-dimensional signal is copied and extended with zeros.
    if new_dimensions.len()==1{
        for i in range(0, old_dimensions[0]){
            new_data[new_offset+i]=old_data[old_offset+i];
        }
        //if this isn't last subsignal add zeros
        if last{
            return old_dimensions[0];
        }
        else{
            return new_dimensions[0];
        }
    }
    //n-dimensional signal is transforned as list of (n-1)-dimensional signals.
    else{
        //size of new signal
        let mut size=0u;
        let mut number_of_old_elements=1u;
        let mut number_of_new_elements=1u;
        //compute number of elements
        for i in range(1, new_dimensions.len()){
            number_of_old_elements*=old_dimensions[i];
            number_of_new_elements*=new_dimensions[i];
        }
        //this is last signal
        if last{
            //transform subsignals except the last one
            for i in range(0, old_dimensions[0]-1){
                size+=transform(
                    old_data, old_offset+i*number_of_old_elements,
                    new_data, new_offset+i*number_of_new_elements,
                    old_dimensions.slice_from(1), new_dimensions.slice_from(1), false
                );
            }
            //transform last subsignal
            size+=transform(
                old_data, old_offset+(old_dimensions[0]-1)*number_of_old_elements,
                new_data, new_offset+(old_dimensions[0]-1)*number_of_new_elements,
                old_dimensions.slice_from(1), new_dimensions.slice_from(1), true
            );
        }
        //this isn't last signal
        else{
            //transform subsignals
            for i in range(0, old_dimensions[0]){
                size+=transform(
                    old_data, old_offset+i*number_of_old_elements,
                    new_data, new_offset+i*number_of_new_elements,
                    old_dimensions.slice_from(1), new_dimensions.slice_from(1), false
                );
            }
            size+=(new_dimensions[0]-old_dimensions[0])*number_of_new_elements;
        }
        return size;
    }
}

/*
    Transforms two n-dimensional signals to one dimensional signals.
    s1: first signal
    s2: second signal
    cut_zeros: cut trailing zeros
    return: transformed signals
*/
fn transform_signals_to_1D(s1:&Signal, s2:&Signal, cut_zeros:bool)->(~Signal, ~Signal){
    let mut new_dimensions:~[uint]=slice::from_elem(s1.dimensions.len(), 0u);
    //compute length of new signals with trailing zeros
    let mut number_of_elements=1u;
    for i in range(0, s1.dimensions.len()){
        new_dimensions[i]=s1.dimensions[i]+s2.dimensions[i]-1;
        number_of_elements*=new_dimensions[i];
    }
    
    //transformation
    let mut new_data1:~[f64]=slice::from_elem(number_of_elements, 0.0);
    let mut new_data2:~[f64]=slice::from_elem(number_of_elements, 0.0);
    let s1_size=transform(s1.data, 0u, new_data1, 0u, s1.dimensions, new_dimensions, cut_zeros);
    let s2_size=transform(s2.data, 0u, new_data2, 0u, s2.dimensions, new_dimensions, cut_zeros);
    new_data1.truncate(s1_size);
    new_data2.truncate(s2_size);
    return (
        ~Signal{
            data:new_data1,
            dimensions:~[s1_size]
        },
        ~Signal{
            data:new_data2,
            dimensions:~[s2_size]
        }
    );
}

/*
    Computes nth element of convolution of two signals.
    n: index of element to compute
    input: input signal
    input_len: length of input signal
    kernel: kernel signal
    kernel_len: length of kernel signal
*/
fn convolve_one_element(n:uint, input:&[f64], input_size:uint, kernel:&[f64], kernel_size:uint)->f64{
    let mut sum=0.0f64;
    let mut i=0u;
    while i<kernel_size{
        if n<i{
            break;
        }
        else if n>=input_size+i{
            i=n-input_size;
        }
        else{
            sum+=input[n-i]*kernel[i];
        }
        i+=1;
    }
    return sum;
}

/*
    Convolves two signals using naive aproach and shared memory.
    input: input signal
    kernel: kernel signal
    dims: dimensions of output signal
    returns: result of convolution signal
*/
fn convolve_with_shared_memory(input:~Signal, kernel:~Signal, dims:~[uint], number_of_workers:uint)->~Signal{
    let size=input.dimensions[0]+kernel.dimensions[0]-1;
    let input_size=input.dimensions[0];
    let kernel_size=kernel.dimensions[0];
    let input_arc=Arc::new(input.data);
    let kernel_arc=Arc::new(kernel.data);
    let mut job_senders:~[Sender<(Action, uint)>]=~[];
    let (result_Sender, result_Receiver):(Sender<(uint, f64)>, Receiver<(uint, f64)>)=channel();
    let mut data:~[f64]=slice::from_elem(size, 0.0);
    
    //spawn worker threads
    for _ in range(0u, number_of_workers){
        let (job_sender, job_receiver):(Sender<(Action, uint)>, Receiver<(Action, uint)>)=channel();
        job_senders.push(job_sender);
        let result_Sender=result_Sender.clone();
        let input_arc=input_arc.clone();
        let kernel_arc=kernel_arc.clone();
        spawn(proc(){
            let input=input_arc.as_slice();
            let kernel=kernel_arc.as_slice();
            loop{
                match job_receiver.recv(){
                    (Compute, n)=>{
                        let value=convolve_one_element(n, input, input_size, kernel, kernel_size);
                        result_Sender.send((n, value));
                    }
                    (Terminate, _)=>{
                        break
                    }
                }
            }
        })
    }
    
    //send work to workers
    for i in range(0u, size){
        job_senders[i%job_senders.len()].send((Compute, i));
    }

    //send terminate to workers
    for sender in job_senders.mut_iter(){
        sender.send((Terminate, 0));
    }

    //receive elements of output signal
    for _ in range(0u, size){
        let (n, value)=result_Receiver.recv();
        data[n]=value;
    }

    return ~Signal{
        data:data,
        dimensions:dims
    };
}

/*
    Convolves two vectors.
    input: first vector
    kernel: second vector
    returns: result of convolution
*/
fn convolve_vectors(input:&[f64], kernel:&[f64])->~[f64]{
    let input_len=input.len();
    let kernel_len=kernel.len();
    return slice::from_fn(input_len+kernel_len-1, |n:uint|{
        convolve_one_element(n, input, input_len, kernel, kernel_len)
    });
}

/*
    Convolves two vectors.
    input: first vector
    kernel: second vector
    returns: result of convolution
*/
fn convolve_vectors_with_padding(input:&[f64], kernel:&[f64], prepend:uint, append:uint)->~[f64]{
    let input_len=input.len();
    let kernel_len=kernel.len();
    let output_len=input_len+append-(kernel_len-1)+prepend;
    let result=slice::from_fn(output_len, |n:uint|{
        convolve_one_element(n+(kernel_len-1)-prepend, input, input_len, kernel, kernel_len)
    });
    return result;
}

/*
    Convolves two signals using naive aproach.
    input: input signal
    kernel: kernel signal
    dims: dimensions of output signal
    returns: result of convolution signal
*/
fn convolve_in_single_thread(input:~Signal, kernel:~Signal, dims:~[uint])->~Signal{
    return ~Signal{
        data:convolve_vectors(input.data, kernel.data),
        dimensions:dims
    };
}

/*
    Gets nth segment of input signal for convolving it with kernel using
    overlap-save method.
    data: signal data
    segment_size: size of segment
    kernel_size: size of kernel
    index: 
*/
fn get_segment(data:&[f64], segment_size:uint, output_size:uint, kernel_size:uint, index:uint)->(~[f64], uint, uint){
    let start=segment_size*index;
    let (start, prepend)=if start<kernel_size-1{
        (0u, kernel_size-1-start)
    }
    else{
        (start-(kernel_size-1), 0u)
    };
    let end=segment_size*(index+1);
    let end=if end>output_size{
        output_size
    }
    else {
        end
    };
    let (end, append)=if end>=data.len(){
        (data.len(), end-data.len())
    }
    else{
        (end, 0u)
    };
    let segment=slice::from_fn(end-start, |n:uint|{data[start+n]});
    return(segment, prepend, append);
}

/*
    Convolves two signals using overlap-save method and message passing.
    input: input signal
    kernel: kernel signal
    dims: dimensions of output signal
    returns: result of convolution signal
*/
fn convolve_by_message_passing(input:~Signal, kernel:~Signal, dims:~[uint], number_of_workers:uint)->~Signal{
    let result_len=input.dimensions[0]+kernel.dimensions[0]-1;
    let segment_size=result_len/number_of_workers+1;
    let (result_sender, result_receiver):(Sender<(uint, ~[f64])>, Receiver<(uint, ~[f64])>)=channel();
    let kernel_dimensions=kernel.dimensions.clone();
    let kernel_arc=Arc::new(kernel.data);
    let mut data:~[~[f64]]=slice::from_elem(number_of_workers, ~[]);
	
	//spawn workers
    for i in range(0u, number_of_workers){
        let result_sender=result_sender.clone();
        let (segment, prepend, append)=get_segment(input.data, segment_size, result_len, kernel_dimensions[0], i);
        let kernel_arc=kernel_arc.clone();
        spawn(proc(){
            let kernel=kernel_arc.as_slice();
            let result=(i, convolve_vectors_with_padding(segment, kernel, prepend, append));
            result_sender.send(result);
        })
    }

    //receive results
    for _ in range(0u, number_of_workers){
        let (n, value)=result_receiver.recv();
        data[n]=value;
    }

    return ~Signal{
        data:data.concat_vec(),
        dimensions:dims
    };
}

/*
    Main function.
*/
fn main(){
    init_devil();
    let args=parse_command_line_arguments();

    if args.help{
        print_help();
        return;
    }

    //load input signal
    let maybe_input=load_signal(args.input_filename, args.input_columns);
    let mut input:~Signal;
    if maybe_input.is_err(){
        println(maybe_input.unwrap_err());
        return;
    }
    else{
        input=maybe_input.unwrap();
    } 

    //load kernel signal
    let maybe_kernel=load_signal(args.kernel_filename, args.kernel_columns);
    let mut kernel:~Signal;
    if maybe_kernel.is_err(){
        println(maybe_kernel.unwrap_err());
        return;
    }
    else{
        kernel=maybe_kernel.unwrap();
    }

    let (input, kernel)=equalize_number_of_dimensions(input, kernel);
    
    //compute input signals dimensions
    let output_dimensions=slice::from_fn(input.dimensions.len(), |i:uint|{
        input.dimensions[i]+kernel.dimensions[i]-1
    });
    
    let (input, kernel)=transform_signals_to_1D(input, kernel, true);
    //convolve signals
    let output=match args.method {
        SingleNaive=>convolve_in_single_thread(input, kernel, output_dimensions),
        MultiNaive=>convolve_with_shared_memory(input, kernel, output_dimensions, args.number_of_workers),
        OverlapSave=>convolve_by_message_passing(input, kernel, output_dimensions, args.number_of_workers)
    };

    let result=save_signal(output, args.output_filename, args.output_columns);
    if result.is_err(){
        println(result.unwrap_err());
    }
}

