//===-- CutEnumeration.cpp - Exp. support for MAPBUF buffer placement -----*-
// C++
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
//===----------------------------------------------------------------------===//

#include "llvm/Support/Path.h"
#include <fstream>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <vector>

#include "experimental/Support/CutEnumeration.h"

using namespace dynamatic::experimental;

void Cuts::readFromFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    llvm::errs() << "Unable to open file: " + filename;
  }

  std::string line;
  std::string node;
  std::string leaf;
  int numberOfCuts = 0;
  int cutSize = 0;

  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    if (line[0] != '\t') {
      std::istringstream iss(line);
      iss >> node >> numberOfCuts;
      for (int i = 0; i < numberOfCuts; i++) {
        std::getline(file, line);
        if (line.substr(0, 5) == "Cut #") {
          Cut newCut(node);
          std::istringstream iss(line);
          std::string dummy;
          iss >> dummy >> dummy >> cutSize;
          for (int j = 0; j < cutSize; ++j) {
            std::getline(file, line);
            leaf = line.substr(1); // Remove leading tab
            newCut.addLeaf(leaf);
          }
          addCut(node, newCut);
        } else {
          llvm::errs() << "No cut found!\n";
          llvm::errs() << line << "\n";
        }
      }
    } else {
      llvm::errs() << "Tab found!\n";
      llvm::errs() << line << "\n";
    }
  }
}

std::set<std::string> BlifData::getPrimaryInputs() const {
  std::set<std::string> primaryInputs;
  primaryInputs.insert(m_inputs.begin(), m_inputs.end());
  primaryInputs.insert(m_latchOutputs.begin(), m_latchOutputs.end());
  primaryInputs.insert(m_constZeroNodes.begin(), m_constZeroNodes.end());
  primaryInputs.insert(m_constOneNodes.begin(), m_constOneNodes.end());
  return primaryInputs;
}

std::set<std::string> BlifData::getNodes() const {
  std::set<std::string> set(m_nodesTopologicalOrder.begin(),
                            m_nodesTopologicalOrder.end());
  return set;
}

std::set<std::string> BlifData::getPrimaryOutputs() const {
  std::set<std::string> primaryOutputs;
  primaryOutputs.insert(m_outputs.begin(), m_outputs.end());
  primaryOutputs.insert(m_latchInputs.begin(), m_latchInputs.end());
  return primaryOutputs;
}

void BlifData::traverseUtil(const std::string &node,
                            std::set<std::string> &visitedNodes) {
  if (std::find(m_nodesTopologicalOrder.begin(), m_nodesTopologicalOrder.end(),
                node) != m_nodesTopologicalOrder.end()) {
    return;
  }

  visitedNodes.insert(node);

  for (const auto &fanin : m_nodeFanins.at(node)) {
    if (std::find(m_nodesTopologicalOrder.begin(),
                  m_nodesTopologicalOrder.end(),
                  fanin) == m_nodesTopologicalOrder.end() &&
        visitedNodes.count(fanin) > 0) {
      // throw std::runtime_error("Cyclic dependency detected!");
    } else if (std::find(m_nodesTopologicalOrder.begin(),
                         m_nodesTopologicalOrder.end(),
                         fanin) == m_nodesTopologicalOrder.end()) {
      traverseUtil(fanin, visitedNodes);
    }
  }

  visitedNodes.erase(node);
  m_nodesTopologicalOrder.push_back(node);
}

void BlifData::traverseNodes() {
  m_primaryInputs.insert(m_inputs.begin(), m_inputs.end()); 
  m_primaryInputs.insert(m_latchOutputs.begin(), m_latchOutputs.end());  
  m_primaryInputs.insert(m_constZeroNodes.begin(), m_constZeroNodes.end());
  m_primaryInputs.insert(m_constOneNodes.begin(), m_constOneNodes.end());

  m_primaryOutputs.insert(m_outputs.begin(), m_outputs.end()); 
  m_primaryOutputs.insert(m_latchInputs.begin(), m_latchInputs.end()); 

  for (const auto &input : m_primaryInputs) {
    m_nodesTopologicalOrder.push_back(input);
  }

  std::set<std::string> visitedNodes;
  for (const auto &node : m_primaryOutputs) {
    traverseUtil(node, visitedNodes);
  }

  for (const auto &node : m_nodesTopologicalOrder) {
    if (m_nodeFanins.find(node) != m_nodeFanins.end()) {
      for (const auto &fanin : m_nodeFanins.at(node)) {
        m_nodeFanouts[fanin].insert(node);
      }
    }
  }
}

void BlifData::printModuleInfo() {
  std::cout << "Module Name: " << m_modulename << std::endl;

  std::cout << "Inputs: ";
  for (const auto &input : getPrimaryInputs()) {
    std::cout << input << " ";
  }
  std::cout << std::endl;

  std::cout << "Outputs: ";
  for (const auto &output : m_outputs) {
    std::cout << output << " ";
  }
  std::cout << std::endl;

  std::cout << "Nodes: ";
  for (const auto &node : m_nodesTopologicalOrder) {
    std::cout << node << " ";
  }
  std::cout << std::endl;

  std::cout << "Fanouts: " << std::endl;
  for (const auto &[node, fanouts] : m_nodeFanouts) {
    std::cout << "Node: " << node << std::endl;
    std::cout << "Fanouts: ";
    for (const auto &fanout : fanouts) {
      std::cout << fanout << " ";
    }
    std::cout << "\n";
  }
}

BlifData BlifParser::parseBlifFile(const std::string &filename) {
  BlifData data;
  std::ifstream file(filename);
  if (!file.is_open()) {
    // throw std::runtime_error("Unable to open file: " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    while (line.back() == '\\') {
      line.pop_back();
      std::string next_line;
      std::getline(file, next_line);
      line += next_line;
    }

    if (line.empty() || line.find("#") == 0) {
      continue;
    }

    if (line.find(".model") == 0) {
      data.setModuleName(line.substr(7));
    }

    else if (line.find(".inputs") == 0) {
      std::string inputs = line.substr(8);
      std::istringstream iss(inputs);
      std::string input;
      while (iss >> input) {
        data.addInput(input);
      }
    }

    else if (line.find(".outputs") == 0) {
      std::string outputs = line.substr(9);
      std::istringstream iss(outputs);
      std::string output;
      while (iss >> output) {
        data.addOutput(output);
      }
    }

    else if (line.find(".latch") == 0) {
      std::string latch = line.substr(7);
      std::istringstream iss(latch);
      std::string regInput;
      std::string regOutput;
      iss >> regInput;
      iss >> regOutput;
      data.addRegInput(regInput);
      data.addRegOutput(regOutput);
      data.addLatch(regInput, regOutput);
    }

    else if (line.find(".names") == 0) {
      std::string fanoutNode;
      std::set<std::string> fanins;
      std::string function;
      std::string node;
      std::vector<std::string> nodes;
      std::string names = line.substr(7);
      std::istringstream iss(names);

      while (iss >> node) { // read the nodes
        nodes.push_back(node);
      }

      fanoutNode = nodes.back();

      std::getline(file, line); // read the function
      iss >> function;

      if (nodes.size() == 1) { // if constant
        if (function == "0") {
          data.addConstZeroNode(node);
        } else if (function == "1") {
          data.addConstOneNode(node);
        } else {
          std::runtime_error("Invalid function for node: " + node);
        }
      }

      for (int i = 0; i < nodes.size() - 1;
           i++) { // add fanins of the fanout node
        fanins.insert(nodes[i]);
      }

      data.addNodes(fanoutNode);
      data.addNodeFanins(fanoutNode, fanins);
      data.addNodeFunctions(fanoutNode, function);
    }

    else if (line.find(".subckt") == 0) {
      std::runtime_error("Not an AIG Graph! .subckt found!");
    }

    else if (line.find(".end") == 0) {
      break;
    }
  }

  data.traverseNodes();
  return data;
}

std::vector<std::string> BlifData::findPath(const std::string& start, const std::string& end){
    std::vector<std::string> path;
    std::set<std::string> visited;
    std::unordered_map<std::string, std::set<std::string>> fanouts = m_nodeFanouts;

    std::function<bool(const std::string&, const std::string&)> dfs = [&](const std::string& node, const std::string& end) {
        if (node == end) {
            path.push_back(node);
            return true;
        }

        visited.insert(node);
        path.push_back(node);

        for (const auto& fanout : m_nodeFanouts[node]) {
            if (visited.count(fanout) == 0) {
                if (dfs(fanout, end)) {
                    return true;
                }
            }
        }

        path.pop_back();
        return false;
    };

    dfs(start, end);
    return path;
}

// CutEnumeration end
