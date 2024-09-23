//===- small.c - A small example for testing -----------------*- C -*-===//
//
// Takes an input, and adds 1 on every iteration
//
//===----------------------------------------------------------------------===//

#include "small.h"
#include "dynamatic/Integration.h"
#include "stdlib.h"

int small(in_int_t idx) {
  int tmp = 0;
  for (unsigned i = 0; i < N; i++)
    idx += 1;
  return idx;
}

int main(void) {
  in_int_t idx = 0;

  CALL_KERNEL(small, idx);
  return 0;
}
