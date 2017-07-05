#!/usr/bin/python3

import sys
import subprocess

NUMBER_OF_KERNELS=100
C_COMMAND="time -p ./c_conv -i ../inputs/lena.jpg -k ../kernels/kernel{0:02}.csv -o ../outputs/output{0:02}.jpg -m {1} -t {2}"
RUST_COMMAND="time -p ./rust_conv -i ../inputs/lena.jpg  -k ../kernels/kernel{0:02}.csv -o ../outputs/output{0:02}.jpg -m {1} -w {2}"
NUMBER_MIN=1
NUMBER_MAX=16
TABLE_EDGE=""
TABLE_HEAD="|{:5}|{:15}|{:10}|{:10}|{:10}|{:10}|"
TABLE_LINE="|{:5}|{:15}|{:10}|{:10.9}|{:10.9}|{:10.9}|"

class Case():
    def __init__(self, lang, method, number):
        self.lang=lang
        self.method=method
        self.number=number
    
    def execute(self):
        sum_real=0.0
        sum_user=0.0
        sum_sys=0.0
        command=""
        if self.lang=="c":
            command=C_COMMAND
        else:
            command=RUST_COMMAND
        for i in range(0, NUMBER_OF_KERNELS):
            time=subprocess.Popen(command.format(i, self.method, self.number).split(" "), stderr=subprocess.PIPE).communicate()[1].split()
            sum_real+=float(time[1])
            sum_user+=float(time[3])
            sum_sys+=float(time[5])
        self.real=sum_real/NUMBER_OF_KERNELS
        self.user=sum_user/NUMBER_OF_KERNELS
        self.sys=sum_sys/NUMBER_OF_KERNELS

def append_c_cases(cases):
    cases.append(Case("c", "single-naive", 1))
    for number in range(NUMBER_MIN, NUMBER_MAX+1):
        cases.append(Case("c", "multi-naive", number))
    cases.append(Case("c", "fourier", 1))

def append_rust_cases(cases):
    cases.append(Case("rust", "single-naive", 1))
    for method in ["multi-naive", "overlap-save"]:
        for number in range(NUMBER_MIN, NUMBER_MAX+1):
            cases.append(Case("rust", method, number))

args=sys.argv[1:]

cases=[]
if args:
    for arg in args:
        arg=arg.split(":")
        if len(arg)==1:
            if arg[0]=="c":
                append_c_cases(cases)
            else:
                append_rust_cases(cases)
        elif len(arg)==2:
            if arg[0]=="c":
                append_c_cases(cases)
            else:
                append_rust_cases(cases)
            cases=list(filter(lambda case: case.method==arg[1], cases))
        else:
            cases.append(Case(arg[0], arg[1], arg[2]))
            
else:
    append_c_cases(cases)
    append_rust_cases(cases)

len_cases=len(cases)
print("00%", end="")
sys.stdout.flush()
for i in range(0, len_cases):
    case=cases[i]
    case.execute()
    print("\b\b\b{:02}%".format(int(((i+1)/len_cases)*100)), end="")
    sys.stdout.flush()

print()
print(TABLE_HEAD.format("lang", "method", "number", "real (ms)", "user (ms)", "sys (ms)"))

for case in cases:
    print(TABLE_LINE.format(case.lang, case.method, case.number, case.real*1000, case.user*1000, case.sys*1000))
print("\a")


