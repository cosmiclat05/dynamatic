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

#include <fstream>
#include <set>
#include <sstream>
#include <queue>
#include <vector>

#include "experimental/Support/BlifReader.h"
#include "llvm/Support/raw_ostream.h"

using namespace dynamatic::experimental;

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
  //llvm::errs() << "Node: " << node << "\n";
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

}

BlifData BlifParser::parseBlifFile(const std::string &filename) {
  BlifData data;
  std::ifstream file(filename);
  if (!file.is_open()) {
    llvm::errs() << "Unable to open file: " << filename << "\n";
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line.find("#") == 0) {
      continue;
    }

    while (line.back() == '\\') {
      line.pop_back();
      std::string nextLine;
      std::getline(file, nextLine);
      line += nextLine;
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
      std::vector<std::string> currNodes;
      std::string names = line.substr(7);
      std::istringstream iss(names);

      std::string node;
      while (iss >> node) { // read the nodes
        currNodes.push_back(node);
      }

      fanoutNode = currNodes.back();

      std::getline(file, line); // read the function
      std::istringstream iss2(line);
      iss2 >> function;

      if (currNodes.size() == 1) { // if constant
        if (function == "0") {
          data.addConstZeroNode(node);
        } else if (function == "1") {
          data.addConstOneNode(node);
        } else {
          llvm::errs() << "Unknown constant value: " << function << "\n";
        }
      }

      for (int i = 0; i < currNodes.size() - 1;
           i++) { // add fanins of the fanout node
        fanins.insert(currNodes[i]);
        data.addNodeFanout(currNodes[i], fanoutNode);
        data.addNodes(currNodes[i]);
      }

      data.addNodes(fanoutNode);
      data.addNodeFanins(fanoutNode, fanins);
      data.addNodeFunctions(fanoutNode, function);
    }

    else if (line.find(".subckt") == 0) {
      continue;
    }

    else if (line.find(".end") == 0) {
      break;
    }
  }

  //necessary for adding .subckt inputs/outputs 
  for (auto& node : data.getAllNodes()){
    auto fanins = data.getFanins(node);
    if (fanins.empty()){
      data.addInput(node);
    }
    auto fanouts = data.getFanouts(node);
    if (fanouts.empty()){
      data.addOutput(node);
    }
  }

  data.traverseNodes();
  return data;
}

// std::vector<std::string> BlifData::findPath(const std::string &start,
//                                             const std::string &end) {
//   std::vector<std::string> path;
//   std::set<std::string> visited;
//   std::unordered_map<std::string, std::set<std::string>> fanouts =
//       m_nodeFanouts;

//   std::function<bool(const std::string &, const std::string &)> dfs =
//       [&](const std::string &node, const std::string &end) {
//         if (node == end) {
//           path.push_back(node);
//           return true;
//         }

//         visited.insert(node);
//         path.push_back(node);

//         for (const auto &fanout : m_nodeFanouts[node]) {
//           if (visited.count(fanout) == 0) {
//             if (dfs(fanout, end)) {
//               return true;
//             }
//           }
//         }

//         path.pop_back();
//         return false;
//       };

//   dfs(start, end);
//   return path;
// }

std::vector<std::string> BlifData::findPath(const std::string &start, const std::string &end) {
    std::queue<std::string> queue;
    std::unordered_map<std::string, std::string> parent;
    std::set<std::string> visited;

    queue.push(start);
    visited.insert(start);

    while (!queue.empty()) {
        std::string current = queue.front();
        queue.pop();

        if (current == end) {
            // Reconstruct the path
            std::vector<std::string> path;
            while (current != start) {
                path.push_back(current);
                current = parent[current];
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            return path;
        }

        for (const auto &fanout : m_nodeFanouts[current]) {
            if (visited.count(fanout) == 0) {
                queue.push(fanout);
                visited.insert(fanout);
                parent[fanout] = current;
            }
        }
    }

    // If no path is found, return an empty vector
    return {};
}

std::set<std::string>
BlifData::findNodesWithLimitedWavyInputs(size_t limit,
                                         std::set<std::string>& wavyLine) {
  std::set<std::string> nodesWithLimitedPrimaryInputs;

  for (const auto &node : m_nodesTopologicalOrder) {
    std::set<std::string> wavyInputs = findWavyInputsOfNode(node, wavyLine);
    if (wavyInputs.size() <= limit) {
      nodesWithLimitedPrimaryInputs.insert(node);
    }
  }
  return nodesWithLimitedPrimaryInputs;
}

std::set<std::string>
BlifData::findWavyInputsOfNode(const std::string &node,
                               std::set<std::string>& wavyLine) {
  std::set<std::string> primaryInputs;
  std::set<std::string> visited;
  std::function<void(const std::string &)> dfs =
      [&](const std::string &currentNode) {
        if (visited.count(currentNode) > 0) {
          return;
        }
        visited.insert(currentNode);

        if (wavyLine.count(currentNode) > 0) {
          primaryInputs.insert(currentNode);
          return;
        }

        for (const auto &fanin : m_nodeFanins[currentNode]) {
          dfs(fanin);
        }
      };

  dfs(node);
  return primaryInputs;
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

// BlifReader end
