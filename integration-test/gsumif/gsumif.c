//===- gsumif.c - Testing long-latency--loop-carried dependency ---*- C -*-===//
//
// Implements a kernel with long-latency--loop-carried dependency in the loop
// body
//
// Author: Jianyi Cheng, DSS
// https://zenodo.org/record/3561115
//
//===---------------------------------------------------------------------===//

#include "dynamatic/Integration.h"
#include "gsumif.h"
#include <stdlib.h>

int gsumif(in_float_t a[1000]) {
  int i;
  int d;
  int s = 0.0;

  for (i = 0; i < 1000; i++) {
    d = a[i];
    if (d >= 0) {
      int p;
      if (i > 5)
        p = ((d + (int)0.25) * d + (int)0.5) * d + (int)0.125;
      else
        p = ((d + (int)0.64) * d + (int)0.7) * d + (int)0.21;
      s += p;
    }
  }
  return s;
}

#define AMOUNT_OF_TEST 1

int main(void) {
  in_float_t a[1000];
  in_float_t b[1000];

  for (int i = 0; i < 1000; ++i) {
    a[i] = (int)1 - i;
    b[i] = (int)i + 10;

    if (i % 100 == 0)
      a[i] = i;
  }

  CALL_KERNEL(gsumif, a);
  return 0;
}
