//===- SubjectGraph.h - Exp. support for MAPBUF buffer placement -------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: Add description.
//
//===-----------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H
#define EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H

#include "BlifReader.h"
#include "dynamatic/Support/LLVM.h"
#include "gurobi_c++.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"

#include <boost/functional/hash/extensions.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

namespace dynamatic {
namespace experimental {



class Module {
  Operation *module;
  SmallVector<Operation *> inputs;
  SmallVector<Operation *> outputs;
  BlifData *blifData;
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H