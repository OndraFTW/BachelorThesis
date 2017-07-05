# Author: Ondřej Šlampa
# Project: Bachelor thesis

C_NAME=c_conv
C_COMPILER=gcc
C_FLAGS=-std=c99 -Wall -Wextra -pedantic -Wfloat-equal -Wundef -Wpointer-arith -lIL -lILU -lm -lfftw3 -fopenmp -g -pg
C_OPTIMIZE_FLAFS=-std=c99 -Wall -Wextra -pedantic -Wfloat-equal -Wundef -Wpointer-arith -lIL -lILU -lm -lfftw3 -fopenmp -O3 -Ofast -march=native

RUST_NAME=rust_conv
RUST_COMPILER=rustc
RUST_FLAGS=-Z verbose
RUST_OPTIMIZE_FLAGS=--opt-level 3

main:
	make co
	make rusto

.PHONY: c cvalg rust rusto cmp

c:
	$(C_COMPILER) conv.c $(C_FLAGS) -o $(C_NAME)

co:
	$(C_COMPILER) conv.c $(C_OPTIMIZE_FLAFS) -o $(C_NAME)

rust:
	$(RUST_COMPILER) conv.rs $(RUST_FLAGS) -o $(RUST_NAME)

rusto:
	$(RUST_COMPILER) conv.rs $(RUST_OPTIMIZE_FLAGS) -o $(RUST_NAME)

