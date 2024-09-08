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
#include "gurobi_c++.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>  // Add this line

using namespace mlir;

namespace dynamatic {
namespace experimental {

class Cut {
public:
  GRBVar cutSelection;
  std::string node;
  std::vector<std::string> leaves;


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
public:
    std::unordered_map<std::string, std::vector<Cut> > cuts;  


    void addCut(const std::string& node, const Cut& newCut){
      if (cuts.find(node) == cuts.end()) {
          cuts[node] = std::vector<Cut>();
      }
      cuts[node].push_back(newCut);
    }
    
    std::vector<Cut> getCuts(const std::string& node){
      if (cuts.find(node) != cuts.end()) {
          return cuts[node];
      }
      return std::vector<Cut>();
    }

    void printCuts(){
      std::ofstream outFile("cuts_output.txt");
      if (!outFile.is_open()) {
          llvm::errs() << "Error: Unable to open file for writing.\n";
          return;
      }

      for (const auto& nodeCuts : cuts) {
          const std::string& node = nodeCuts.first;
          const std::vector<Cut>& cutList = nodeCuts.second;
          
          outFile << "Node: " << node << "\n";
          
          for (size_t i = 0; i < cutList.size(); ++i) {
              const Cut& cut = cutList[i];
              outFile << "  Cut #" << i << ":\n";
              
              const std::vector<std::string>& leaves = cut.getLeaves();
              for (const auto& leaf : leaves) {
                  outFile << "    " << leaf << "\n";
              }
          }
          
          outFile << "\n";
      }

      outFile.close();
    }
    
    // Method to read from file and populate cuts
    void readFromFile(const std::string& filename);
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H
