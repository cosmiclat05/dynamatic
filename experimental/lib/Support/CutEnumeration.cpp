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
#include <list>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "experimental/Support/CutEnumeration.h"
#include "llvm/Support/raw_ostream.h"

using namespace dynamatic::experimental;

void Cuts::readFromFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    llvm::errs() << "Unable to open file: " + filename;
  }

  std::string line;
  std::string nodeName;
  std::string leaf;
  int numberOfCuts = 0;
  int cutSize = 0;

  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    if (line[0] != '\t') {
      std::istringstream iss(line);
      iss >> nodeName >> numberOfCuts;
      for (int i = 0; i < numberOfCuts; i++) {
        std::getline(file, line);
        if (line.substr(0, 5) == "Cut #") {
          Node *node = blif.getNodeByName(nodeName);
          Cut newCut(node);
          std::istringstream iss(line);
          std::string dummy;
          iss >> dummy >> dummy >> cutSize;
          for (int j = 0; j < cutSize; ++j) {
            std::getline(file, line);
            leaf = line.substr(1); // Remove leading tab
            newCut.addLeaf(blif.getNodeByName(leaf));
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

template <typename T>
bool isSubset(const std::set<T> &setA, const std::set<T> &setB) {
  if (setA.size() > setB.size()) {
    return false; // A cannot be a subset if it's larger than B
  }

  for (const T &element : setA) {
    if (setB.find(element) == setB.end()) {
      return false; // Element from A not found in B
    }
  }

  return true; // All elements of A were found in B
}

std::vector<Cut>
Cuts::enumerateCuts(Node *node, const std::vector<std::vector<Cut>> &faninCuts,
                    int lutSize) {

  std::list<Cut> cutsList;
  std::vector<Cut> cuts1 = faninCuts[0];
  std::vector<Cut> cuts2 = faninCuts[1];

  int depth = 0;
  int maxDepth = 0;

  for (auto &cut1 : cuts1) {
    std::list<Cut> localCuts;
    for (auto &cut2 : cuts2) {
      std::set<Node *> leaves1 = cut1.getLeaves();
      std::set<Node *> leaves2 = cut2.getLeaves();
      // get max depth of cut1 and cut2
      depth = std::max(cut1.getDepth(), cut2.getDepth());
      if (depth > maxDepth) {
        maxDepth = depth;
      }
      std::set<Node *> merge = leaves1;
      merge.insert(leaves2.begin(), leaves2.end());
      Cut mergedCut(node, merge, depth);
      if (mergedCut.getLeaves().size() <= lutSize + maxExpansion) {
        localCuts.push_back(mergedCut);
      }
    }
    { cutsList.splice(cutsList.end(), localCuts); }
  }

  std::vector<Cut> cuts(cutsList.begin(), cutsList.end());

  std::vector<Cut> cutsPruned;
  for (auto &cut : cuts) {
    bool isSubsetOfAnother = false;
    for (auto &otherCut : cuts) {
      if (cut.getLeaves() == otherCut.getLeaves()) {
        continue;
      }
      if (isSubset(cut.getLeaves(), otherCut.getLeaves())) {
        isSubsetOfAnother = true;
        break;
      }
    }
    if (!isSubsetOfAnother) {
      cutsPruned.push_back(cut);
    }
  }

  std::sort(cutsPruned.begin(), cutsPruned.end(), [](Cut &a, Cut &b) {
    if (a.getDepth() == b.getDepth()) {
      return a.getLeaves().size() > b.getLeaves().size();
    }
    return a.getDepth() > b.getDepth();
  });

  std::vector<Cut> cutsExceedingLutSize;
  for (auto &cut : cutsPruned) {
    if (cut.getLeaves().size() > lutSize) {
      cutsExceedingLutSize.push_back(cut);
    }
  }

  std::vector<Cut> cutsWithLimitedLeaves;
  for (auto &cut : cutsPruned) {
    if (cut.getLeaves().size() <= lutSize) {
      cutsWithLimitedLeaves.push_back(cut);
    }
  }

  std::vector<Cut> cutsMerged;
  // Keep only the first 20 biggest cuts
  for (size_t i = 0; i < cutsWithLimitedLeaves.size() && i < 30; ++i) {
    cutsMerged.push_back(cutsWithLimitedLeaves[i]);
  }

  for (size_t i = 0; i < cutsExceedingLutSize.size() && i < 10; ++i) {
    cutsMerged.push_back(cutsExceedingLutSize[i]);
  }

  cutsMerged.erase(std::unique(cutsMerged.begin(), cutsMerged.end(),
                               [](Cut &a, Cut &b) {
                                 return a.getLeaves() == b.getLeaves();
                               }),
                   cutsMerged.end());

  // Add trivial cut
  cutsMerged.emplace_back(node, node, maxDepth + 1);

  return cutsMerged;
}

std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>
Cuts::cutless() {
  int n = 0;
  std::set<Node *> currentWavyLine = blif.getPrimaryInputs();
  std::set<Node *> nextWavyLine;
  std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>
      cutlessCuts;

  while (true) {
    nextWavyLine = blif.findNodesWithLimitedWavyInputs(6, currentWavyLine);
    if (nextWavyLine.size() == currentWavyLine.size()) {
      break;
    }

    for (auto *node : nextWavyLine) {
      cutlessCuts[node].emplace_back(
          node, blif.findWavyInputsOfNode(node, currentWavyLine), n);
    }

    n++;
    // currentWavyLine = nextWavyLine;
    currentWavyLine.insert(nextWavyLine.begin(), nextWavyLine.end());
    llvm::errs() << n << " " << currentWavyLine.size() << "\n";
  }

  for (auto &[node, cuts] : cutlessCuts) {
    cuts.erase(std::unique(cuts.begin(), cuts.end(),
                           [](Cut &a, Cut &b) {
                             const auto &leavesA = a.getLeaves();
                             const auto &leavesB = b.getLeaves();

                             // Compare the sizes first as an optimization
                             if (leavesA.size() != leavesB.size()) {
                               return false;
                             }

                             // Compare elements in the set based on Node's name
                             // strings
                             return std::equal(
                                 leavesA.begin(), leavesA.end(),
                                 leavesB.begin(), leavesB.end(),
                                 [](const Node *nodeA, const Node *nodeB) {
                                   return nodeA->getName() == nodeB->getName();
                                 });
                           }),
               cuts.end());
  }

  return cutlessCuts;
}

// Usage in your main cut enumeration function
std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>
Cuts::computeAllCuts() {
  // cuts = cutless();
  std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>
      cutResults;

  for (auto *node : blif.getNodesInOrder()) {
    if (node->isPrimaryInput()) {
      cutResults[node].emplace_back(node, node, 0);
      continue;
    }

    auto fanins = node->getFanins();

    if (fanins.size() == 1) {
      for (auto &cuts : cutResults[*fanins.begin()]) {
        cutResults[node].emplace_back(node, cuts.getLeaves(), cuts.getDepth());
      }
      continue;
    }

    if (fanins.size() == 2) {
      std::set<Node *> insOfFanin1 = (*fanins.begin())->getFanins();
      std::set<Node *> insOfFanin2 =
          (*std::next(fanins.begin(), 1))->getFanins();

      if ((insOfFanin1 == insOfFanin2) && (!insOfFanin1.empty())) {
        for (auto &cuts : cutResults[*fanins.begin()]) {
          cutResults[node].emplace_back(node, cuts.getLeaves(),
                                        cuts.getDepth());
        }

        int maxDepth = 0;
        for (auto &cut : cutResults[*fanins.begin()]) {
          if (cut.getDepth() > maxDepth) {
            maxDepth = cut.getDepth();
          }
        }
        cutResults[node].emplace_back(node, node, maxDepth + 1);
        continue;
      }
    }

    std::vector<std::vector<Cut>> faninCuts;
    for (auto &fanin : fanins) {
      faninCuts.push_back(cutResults[fanin]);
    }

    std::vector<Cut> cutsEnumerated = enumerateCuts(node, faninCuts, lutSize);
    if (cutsEnumerated.size() == 1 && cutsEnumerated[0].getLeaves().empty()) {
      continue;
    }

    cutResults[node].insert(cutResults[node].end(), cutsEnumerated.begin(),
                            cutsEnumerated.end());
  }

  for (auto &[node, cuts] : cutResults) {

    std::sort(cuts.begin(), cuts.end(), [](Cut &a, Cut &b) {
      if (a.getDepth() == b.getDepth()) {
        return a.getLeaves().size() < b.getLeaves().size();
      }
      return a.getDepth() < b.getDepth();
    });

    cuts.erase(std::remove_if(cuts.begin(), cuts.end(),
                              [this](Cut &cut) {
                                return cut.getLeaves().size() > lutSize;
                              }),
               cuts.end());

    cuts.erase(std::unique(cuts.begin(), cuts.end(),
                           [](Cut &a, Cut &b) {
                             return a.getLeaves() == b.getLeaves();
                           }),
               cuts.end());
  }

  return cutResults;
}

void Cuts::runCutAlgos(bool computeAllCutsBool, bool cutlessBool,
                       bool cutlessRealBool, bool anchors) {

  std::unordered_map<Node *, std::vector<Cut>, NodePtrHash, NodePtrEqual>
      tempCuts;

  if (computeAllCutsBool) {
    auto cutsTemp1 = computeAllCuts();
    for (auto &pair : cutsTemp1) {
      Node *node = pair.first;
      std::vector<Cut> &cutVector = pair.second;
      tempCuts[node].insert(tempCuts[node].end(), cutVector.begin(),
                            cutVector.end());
    }
  }
  if (cutlessBool) {
    auto cutsTemp1 = cutless();
    for (auto &pair : cutsTemp1) {
      Node *node = pair.first;
      std::vector<Cut> &cutVector = pair.second;
      tempCuts[node].insert(tempCuts[node].end(), cutVector.begin(),
                            cutVector.end());
    }
  }
  // if (cutlessRealBool){
  //   auto cutsTemp1 = cutlessReal();
  //   for (auto &pair : cutsTemp1) {
  //     Node* node = pair.first;
  //     std::vector<Cut> &cutVector = pair.second;
  //     tempCuts[node].insert(tempCuts[node].end(), cutVector.begin(),
  //     cutVector.end());
  //   }
  // }

  if (!anchors) {
    for (auto &pair : tempCuts) {
      Node *node = pair.first;
      std::vector<Cut> &cutVector = pair.second;
      cuts[node].insert(cuts[node].end(), cutVector.begin(), cutVector.end());
    }
  } else {
    for (auto &pair : tempCuts) {
      std::vector<std::string> namesToRemove = {
          "__data_anchors_out", "__data_anchors_in", "__anchors_out",
          "__anchors_in"};
      auto removeAnchorsUtil = [namesToRemove](Node *node) {
        for (auto &nameToRemove : namesToRemove) {
          size_t pos = node->str().find(nameToRemove);
          if (pos != std::string::npos) {
            node->setName(
                node->str().erase(pos, std::string(nameToRemove).length()));
            break;
          }
        }
      };

      Node *node = pair.first;
      std::vector<Cut> &cutVector = pair.second;
      removeAnchorsUtil(node);

      for (auto &cut : cutVector) {
        Node *root = cut.root;
        removeAnchorsUtil(root);
        std::set<Node *> leaves = cut.leaves;
        for (const auto &leaf : leaves) {
          removeAnchorsUtil(leaf);
        }
      }

      cuts[node].insert(cuts[node].end(), cutVector.begin(), cutVector.end());
    }
  }

  for (auto &[node, cutVector] : cuts) {
    std::sort(cutVector.begin(), cutVector.end(),
              [](const Cut &a, const Cut &b) {
                // Get the sets first
                const auto leavesA = a.getLeaves();
                const auto leavesB = b.getLeaves();

                // First compare by size
                if (leavesA.size() != leavesB.size()) {
                  return leavesA.size() < leavesB.size();
                }

                // Compare elements using set's iterators
                auto itA = leavesA.begin();
                auto itB = leavesB.begin();
                while (itA != leavesA.end() && itB != leavesB.end()) {
                  const Node *nodeA = *itA;
                  const Node *nodeB = *itB;

                  // Handle null pointers
                  if (!nodeA || !nodeB) {
                    if (nodeA == nodeB) {
                      ++itA;
                      ++itB;
                      continue;
                    }
                    return !nodeA;
                  }

                  // Compare names
                  if (nodeA->getName() != nodeB->getName()) {
                    return nodeA->getName() < nodeB->getName();
                  }

                  ++itA;
                  ++itB;
                }
                return false; // Sets are equal
              });

    // Remove duplicates
    cutVector.erase(std::unique(cutVector.begin(), cutVector.end(),
                                [](const Cut &a, const Cut &b) {
                                  const auto leavesA = a.getLeaves();
                                  const auto leavesB = b.getLeaves();

                                  if (leavesA.size() != leavesB.size()) {
                                    return false;
                                  }

                                  // Compare sets
                                  auto itA = leavesA.begin();
                                  auto itB = leavesB.begin();
                                  while (itA != leavesA.end() &&
                                         itB != leavesB.end()) {
                                    const Node *nodeA = *itA;
                                    const Node *nodeB = *itB;

                                    // Handle null pointers
                                    if (!nodeA || !nodeB) {
                                      if (nodeA != nodeB) {
                                        return false;
                                      }
                                      ++itA;
                                      ++itB;
                                      continue;
                                    }

                                    // Compare names
                                    if (nodeA->getName() != nodeB->getName()) {
                                      return false;
                                    }

                                    ++itA;
                                    ++itB;
                                  }
                                  return true; // Sets are equal
                                }),
                    cutVector.end());
  }
}

void Cuts::printCuts(std::string filename) {
  std::ofstream outFile("../mapbuf/" + filename);
  if (!outFile.is_open()) {
    llvm::errs() << "Error: Unable to open file for writing.\n";
    return;
  }

  for (auto &nodeCuts : cuts) {
    Node *node = nodeCuts.first;
    std::vector<Cut> &cutList = nodeCuts.second;
    std::size_t numCuts = cutList.size();

    outFile << node->str() << " " << numCuts << "\n";

    for (size_t i = 0; i < cutList.size(); ++i) {
      Cut &cut = cutList[i];
      std::set<Node *> leaves = cut.getLeaves();
      std::size_t cutSize = leaves.size();
      outFile << "Cut #" << i << ": " << cutSize << " depth: " << cut.getDepth()
              << "\n";

      for (auto leaf : leaves) {
        outFile << '\t' << leaf->str() << "\n";
      }
    }
  }
  outFile.close();
}

// CutEnumeration end
