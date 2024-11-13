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

#include "BlifReader.h"
#include "dynamatic/Support/LLVM.h"
#include "gurobi_c++.h"
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
  Node *root;
  std::set<Node *> leaves;
  int depth;

  Cut(Node *root, int depth = 0) : root(root), depth(depth){};
  Cut(Node *root, Node *leaf, int depth = 0)
      : root(root), leaves({leaf}), depth(depth){}; // for trivial cuts
  Cut(Node *root, std::set<Node *> leaves, int depth = 0)
      : root(root), leaves(leaves), depth(depth){};

  void addLeaf(Node *leaf) { leaves.insert(leaf); }

  void addLeaves(std::set<Node *> &leaves) {
    this->leaves.insert(leaves.begin(), leaves.end());
  }

  void setLeaves(std::set<Node *> &leaves) { this->leaves = leaves; }

  Node *getNode() { return root; }

  std::set<Node *> getLeaves() const { return leaves; }

  Node *getRoot() { return root; }

  int getDepth() { return depth; }
};

struct NodePtrHash {
  std::size_t operator()(const Node *node) const {
    return std::hash<std::string>()(node->getName()); // Hash the name
  }
};

struct NodePtrEqual {
  bool operator()(const Node *lhs, const Node *rhs) const {
    return lhs->getName() == rhs->getName();
  }
};

class Cuts {
private:
  experimental::BlifData *blif;
  int lutSize{};
  const int expansionWithChannels = 6;

public:
  static inline std::unordered_map<Node *, std::vector<Cut>, NodePtrHash,
                                   NodePtrEqual>
      cuts;

  Cuts(BlifData *blif, int lutSize) : blif(blif), lutSize(lutSize) {
    this->runCutAlgos();
  };

  std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>
  cutless(bool includeChannels);
  void runCutAlgos();
  static void printCuts(std::string filename);
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H
