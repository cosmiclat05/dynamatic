//===- CutEnumeration.h - Exp. support for MAPBUF buffer placement -------*- C++
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

#ifndef EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H
#define EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H

#include "dynamatic/Support/LLVM.h"
#include "gurobi_c++.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "BlifReader.h"

#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

namespace dynamatic {
namespace experimental {

class Cut {
public:
  GRBVar nodeVar;
  GRBVar cutSelection;
  std::string root;
  std::set<std::string> leaves;
  int depth;

  Cut(std::string root, int depth = 0) : root(root), depth(depth) {};
  Cut(std::string root, std::string leaf, int depth = 0) : root(root), leaves(std::set<std::string>{leaf}), depth(depth) {}; //for trivial cuts
  Cut(std::string root, std::set<std::string> leaves, int depth = 0) : root(root), leaves(leaves), depth(depth) {};

  void addLeaf(const std::string &leaf) { leaves.insert(leaf); }

  const std::string &getNode() const { return root; }

  const std::set<std::string> &getLeaves() const { return leaves; }

  void setLeaves(std::set<std::string>& leaves){
      this->leaves = leaves;
  }

  void addLeaves(std::set<std::string>& leaves){
      this->leaves.insert(leaves.begin(), leaves.end());
  }

  std::string getRoot() const{
      return root;
  }

  int getDepth() const{
      return depth;
  }
};

class Cuts {
public:
  static inline std::unordered_map<std::string, std::vector<Cut>> cuts;
  experimental::BlifData blif;
  int lutSize{};
  int maxExpansion{};

  Cuts(BlifData& blif, int lutSize, int maxExpansion) : blif(blif), lutSize(lutSize), maxExpansion(maxExpansion){};
  
  std::vector<Cut> enumerateCuts(const std::string& node, const std::vector<std::vector<Cut>>& faninCuts, int lutSize);
  std::unordered_map<std::string, std::vector<Cut>> computeAllCuts();
  std::unordered_map<std::string, std::vector<Cut>> cutless();
  std::unordered_map<std::string, std::vector<Cut>> cutlessReal();

  void runCutAlgos(bool computeAllCuts, bool cutless, bool cutlessReal, bool anchors);
  void readFromFile(const std::string &filename);
  static void printCuts(std::string filename);

  void addCut(const std::string &node, const Cut &newCut) {
    if (cuts.find(node) == cuts.end()) {
      cuts[node] = std::vector<Cut>();
    }
    cuts[node].push_back(newCut);
  }

  std::vector<Cut> getCuts(const std::string &node) {
    if (cuts.find(node) != cuts.end()) {
      return cuts[node];
    }
    return std::vector<Cut>();
  }
 
  // Method to read from file and populate cuts
  
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H
