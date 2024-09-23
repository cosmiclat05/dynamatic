//===- small.h - A small example for testing -----------------*- C -*-===//
//
// Takes an input, and adds 1 on every iteration
//
//===----------------------------------------------------------------------===//

#ifndef SMALL_SMALL_H
#define SMALL_SMALL_H

#define N 1000
#define N_DEC 999 // = N - 1

typedef int in_int_t;

/// Computes the finite impulse response between two arrays.
int small(in_int_t idx);

#endif // SMALL_SMALL_H
