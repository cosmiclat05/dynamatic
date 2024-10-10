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

#ifndef EXPERIMENTAL_SUPPORT_BLIF_READER_H
#define EXPERIMENTAL_SUPPORT_BLIF_READER_H

#include "dynamatic/Support/LLVM.h"
#include "gurobi_c++.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"

#include <fstream> // Add this line
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

namespace dynamatic {
namespace experimental {

class BlifData {
private:
  std::string m_modulename;
  std::set<std::string> m_inputs;
  std::set<std::string> m_outputs;
  std::set<std::string> nodes;
  std::set<std::string> m_latchInputs;
  std::set<std::string> m_latchOutputs;
  std::unordered_map<std::string, std::string> m_latches;
  std::set<std::string> m_constZeroNodes;
  std::set<std::string> m_constOneNodes;

  std::unordered_map<std::string, std::set<std::string>> m_nodeFanins;
  std::unordered_map<std::string, std::set<std::string>> m_nodeFanouts;
  std::unordered_map<std::string, std::string> m_nodeFunctions;

  std::vector<std::string> m_nodesTopologicalOrder;

  std::unordered_map<std::string, std::set<std::string>> m_submodules;

  std::set<std::string> m_primaryInputs;
  std::set<std::string> m_primaryOutputs;

public:
  BlifData() = default;

  void setModuleName(const std::string &moduleName) {
    m_modulename = moduleName;
  }

  void addInput(const std::string &input) { m_inputs.insert(input); }

  void addOutput(const std::string &output) { m_outputs.insert(output); }

  void addNodes(const std::string &node) { nodes.insert(node); }

  void addRegInput(const std::string &latchInput) {
    m_latchInputs.insert(latchInput);
  }

  void addRegOutput(const std::string &latchOutput) {
    m_latchOutputs.insert(latchOutput);
  }

  void addLatch(const std::string &latchInput, const std::string &latchOutput) {
    m_latches[latchInput] = latchOutput;
  }

  void addConstZeroNode(const std::string &node) {
    m_constZeroNodes.insert(node);
  }

  void addConstOneNode(const std::string &node) {
    m_constOneNodes.insert(node);
  }

  void addNodeFanins(const std::string &node,
                     const std::set<std::string> &fanins) {
    m_nodeFanins[node] = fanins;
  }

  void addNodeFanouts(const std::string &node,
                      const std::set<std::string> &fanouts) {
    m_nodeFanouts[node] = fanouts;
  }

  void addNodeFanout(const std::string &node,
                     const std::string &fanout) {
    m_nodeFanouts[node].insert(fanout);
  }

  void addNodeFunctions(const std::string &node, const std::string &function) {
    m_nodeFunctions[node] = function;
  }

  void addSubmodule(const std::string &submodule,
                    const std::set<std::string> &signals) {
    m_submodules[submodule] = signals;
  }

  bool isPrimaryInput(const std::string &input) const {
    return m_primaryInputs.count(input) > 0;
  }

  bool isPrimaryOutput(const std::string &output) const {
    return m_primaryOutputs.count(output) > 0;
  }

  bool isInput(const std::string &input) const {
    return m_inputs.count(input) > 0;
  }

  bool isOutput(const std::string &output) const {
    return m_outputs.count(output) > 0;
  }

  bool isRegInput(const std::string &regInput) const {
    return m_latchInputs.count(regInput) > 0;
  }

  bool isRegOutput(const std::string &regOutput) const {
    return m_latchOutputs.count(regOutput) > 0;
  }

  void printModuleInfo();

  void traverseNodes();

  void traverseUtil(const std::string &node,
                    std::set<std::string> &visitedNodes);

  std::set<std::string> getPrimaryInputs() const;

  std::set<std::string> getPrimaryOutputs() const;

  std::set<std::string> getNodes() const;

  std::set<std::string> getAllNodes() const{
    return nodes;
  }

  std::set<std::string> getFanouts(const std::string &node) const {
    if (m_nodeFanouts.count(node) > 0) {
      return m_nodeFanouts.at(node);
    } else {
      return std::set<std::string>();
    }
  }

  std::set<std::string> getFanins(const std::string &node) const {
    if (m_nodeFanins.count(node) > 0) {
      return m_nodeFanins.at(node);
    } 
    else {
      return std::set<std::string>();
    }
  }
  
  std::vector<std::string> findPath(const std::string &start,
                                    const std::string &end);

  std::vector<std::string> getNodesInOrder() const {
    return m_nodesTopologicalOrder;
  }

  std::set<std::string> findNodesWithLimitedWavyInputs(size_t limit, std::set<std::string>& wavyLine);

  std::set<std::string> findWavyInputsOfNode(const std::string &node, std::set<std::string>& wavyLine);
};


class BlifParser {
public:
  BlifParser() = default;
  experimental::BlifData parseBlifFile(const std::string &filename);
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_CUT_ENUMERATION_H
