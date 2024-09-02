//===- CutEnumeration.h - Exp. support for MAPBUF buffer placement -------*- C++ -*-===//
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

#ifndef EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H
#define EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include <string>
#include <vector>

using namespace mlir;

namespace dynamatic {
namespace experimental {

class Cut {
private:
    std::string node;
    std::vector<std::string> leaves;

public:
    Cut(const std::string& n) : node(n) {}
    
    void addLeaf(const std::string& leaf) {
        leaves.push_back(leaf);
    }
    
    const std::string& getNode() const {
        return node;
    }
    
    const std::vector<std::string>& getLeaves() const {
        return leaves;
    }
};

class Cuts {
private:
    std::unordered_map<std::string, Cut> cuts;

public:
    void addCut(const std::string& node, const std::vector<std::string>& leaves);
    
    const Cut* getCut(const std::string& node) const;
    
    // Method to read from file and populate cuts
    void readFromFile(const std::string& filename);
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H
