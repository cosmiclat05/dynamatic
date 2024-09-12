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
#include <set>
#include <vector>
#include <unordered_map>
#include <fstream>  // Add this line

using namespace mlir;

namespace dynamatic {
namespace experimental {

class BlifData{
private:
    std::string m_modulename;
    std::set<std::string> m_inputs;
    std::set<std::string> m_outputs;
    std::set<std::string> nodes;
    std::set<std::string> m_latchInputs;
    std::set<std::string> m_latchOutputs;
    std::map<std::string, std::string> m_latches;
    std::set<std::string> m_constZeroNodes;
    std::set<std::string> m_constOneNodes;

    std::vector<std::string> m_signals;
    std::map<std::string, std::set<std::string>> m_nodeFanins;
    std::map<std::string, std::set<std::string>> m_nodeFanouts;
    std::map<std::string, std::string> m_nodeFunctions;

    std::vector<std::string> m_nodesTopologicalOrder;

    std::map<std::string, std::set<std::string>> m_submodules;

public:
    BlifData() = default;

    void setModuleName(const std::string& moduleName) {
        m_modulename = moduleName;
    }

    void addInput(const std::string& input) {
        m_inputs.insert(input);
    }

    void addOutput(const std::string& output) {
        m_outputs.insert(output);
    }

    void addNodes(const std::string& node) {
        nodes.insert(node);
    }

    void addRegInput(const std::string& latchInput) {
        m_latchInputs.insert(latchInput);
    }

    void addRegOutput(const std::string& latchOutput) {
        m_latchOutputs.insert(latchOutput);
    }

    void addLatch(const std::string& latchInput, const std::string& latchOutput) {
        m_latches[latchInput] = latchOutput;
    }

    void addConstZeroNode(const std::string& node) {
        m_constZeroNodes.insert(node);
    }

    void addConstOneNode(const std::string& node) {
        m_constOneNodes.insert(node);
    }

    void addSignal(const std::string& signal) {
        m_signals.push_back(signal);
    }

    void addNodeFanins(const std::string& node, const std::set<std::string>& fanins) {
        m_nodeFanins[node] = fanins;
    }

    void addNodeFanouts(const std::string& node, const std::set<std::string>& fanouts) {
        m_nodeFanouts[node] = fanouts;
    }

    void addNodeFunctions(const std::string& node, const std::string& function) {
        m_nodeFunctions[node] = function;
    }

    void addSubmodule(const std::string& submodule, const std::set<std::string>& signals) {
        m_submodules[submodule] = signals;
    }

    bool isInput(const std::string& input) const {
        return m_inputs.count(input) > 0;
    }

    bool isOutput(const std::string& output) const {
        return m_outputs.count(output) > 0;
    }

    bool isRegInput(const std::string& regInput) const {
        return m_latchInputs.count(regInput) > 0;
    }

    bool isRegOutput(const std::string& regOutput) const {
        return m_latchOutputs.count(regOutput) > 0;
    }

    void printModuleInfo();

    void traverseNodes();

    void traverseUtil(const std::string& node, std::set<std::string>& visitedNodes);

    std::set<std::string> getPrimaryInputs() const;

    std::set<std::string> getPrimaryOutputs() const;

    std::vector<std::string> getNodesInOrder() const;

};

class BlifParser{ 
public:
    BlifParser() = default;
    BlifData parseBlifFile(const std::string& filename);
};

class Cut {
public:
  GRBVar nodeVar;
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
