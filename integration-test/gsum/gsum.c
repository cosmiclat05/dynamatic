//===- gsum.c - Testing long-latency--loop-carried dependency --*- C -*-===//
//
// Implements a kernel with long-latency--loop-carried dependency in the loop
// body
//
// Author: Jianyi Cheng
// https://zenodo.org/record/3561115
//
//===------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include "gsum.h"

int gsum(in_float_t a[N]) {
  int i;
  int d;
  int s = 0.0;

  for (i = 0; i < N; i++) {
    d = a[i];
    if (d >= 0)
      // An if condition in the loop causes irregular computation.  Static
      // scheduler reserves time slot for each iteration causing unnecessary
      // pipeline stalls.

      s += (((((d + (int)0.64) * d + (int)0.7) * d + (int)0.21) * d +
             (int)0.33) *
            d);
  }
  return s;
}

int main(void) {
  in_float_t a[N];
  in_float_t b[N];

  for (int i = 0; i < N; ++i) {
    a[i] = (int)1 - i;
    b[i] = (int)i + 10;

    if (i % 100 == 0)
      a[i] = i;
  }

  CALL_KERNEL(gsum, a);
  return 0;
}
