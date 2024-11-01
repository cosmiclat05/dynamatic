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

  Node *getNode() { return root; }

  std::set<Node *> getLeaves() const { return leaves; }

  void setLeaves(std::set<Node *> &leaves) { this->leaves = leaves; }

  void addLeaves(std::set<Node *> &leaves) {
    this->leaves.insert(leaves.begin(), leaves.end());
  }

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
public:
  static inline std::unordered_map<Node *, std::vector<Cut>, NodePtrHash,
                                   NodePtrEqual>
      cuts;
  experimental::BlifData* blif;
  int lutSize{};
  int maxExpansion{};

  Cuts(BlifData *blif, int lutSize, int maxExpansion)
      : blif(blif), lutSize(lutSize), maxExpansion(maxExpansion){};

  std::vector<Cut> enumerateCuts(Node *node,
                                 const std::vector<std::vector<Cut>> &faninCuts,
                                 int lutSize);
  std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>
  computeAllCuts();
  std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>
  cutless();
  std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>
  cutlessReal();
  std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>
  cutlessChannels();
  void runCutAlgos(bool computeAllCuts, bool cutless, bool cutlessChannelsBool);
  void readFromFile(const std::string &filename);
  static void printCuts(std::string filename);

  void addCut(Node *node, const Cut &newCut) {
    if (cuts.find(node) == cuts.end()) {
      cuts[node] = std::vector<Cut>();
    }
    cuts[node].push_back(newCut);
  }

  std::vector<Cut> getCuts(Node *node) {
    if (cuts.find(node) != cuts.end()) {
      return cuts[node];
    }
    return std::vector<Cut>();
  }
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H
