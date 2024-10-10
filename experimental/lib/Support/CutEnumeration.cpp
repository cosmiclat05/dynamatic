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
Cuts::enumerateCuts(const std::string &node,
                    const std::vector<std::vector<Cut>> &faninCuts,
                    int lutSize) {

  std::list<Cut> cutsList;
  std::vector<Cut> cuts1 = faninCuts[0];
  std::vector<Cut> cuts2 = faninCuts[1];

  int depth = 0;
  int maxDepth = 0;

  for (const auto &cut1 : cuts1) {
    std::list<Cut> localCuts;
    for (const auto &cut2 : cuts2) {
      std::set<std::string> leaves1 = cut1.getLeaves();
      std::set<std::string> leaves2 = cut2.getLeaves();
      // get max depth of cut1 and cut2
      depth = std::max(cut1.getDepth(), cut2.getDepth());
      if (depth > maxDepth) {
        maxDepth = depth;
      }
      std::set<std::string> merge = leaves1;
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
  for (const auto &cut : cuts) {
    bool isSubsetOfAnother = false;
    for (const auto &otherCut : cuts) {
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

  std::sort(cutsPruned.begin(), cutsPruned.end(),
            [](const Cut &a, const Cut &b) {
              if (a.getDepth() == b.getDepth()) {
                return a.getLeaves().size() > b.getLeaves().size();
              }
              return a.getDepth() > b.getDepth();
            });

  std::vector<Cut> cutsExceedingLutSize;
  for (const auto &cut : cutsPruned) {
    if (cut.getLeaves().size() > lutSize) {
      cutsExceedingLutSize.push_back(cut);
    }
  }

  std::vector<Cut> cutsWithLimitedLeaves;
  for (const auto &cut : cutsPruned) {
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
                               [](const Cut &a, const Cut &b) {
                                 return a.getLeaves() == b.getLeaves();
                               }),
                   cutsMerged.end());

  // Add trivial cut
  cutsMerged.emplace_back(node, node, maxDepth + 1);

  return cutsMerged;
}

std::unordered_map<std::string, std::vector<Cut>> Cuts::cutlessReal(){

  std::unordered_map<std::string, std::vector<Cut>> cutlessCuts;
  std::set<std::string> primaryInputs = blif.getPrimaryInputs();
  for (auto& pi : primaryInputs) {
    cutlessCuts[pi].emplace_back(pi, pi, 0);
  }

  for (const auto &node : blif.getNodesInOrder()) {
    if (blif.isPrimaryInput(node)) {
      continue;
    }

    auto fanins = blif.getFanins(node);

    if (fanins.size() == 1){
      cutlessCuts[node] = cutlessCuts[*fanins.begin()];
    }

    else if (fanins.size() == 2){
      auto node0 = *fanins.begin();
      auto node1 = *std::next(fanins.begin(), 1);
      auto levelNode0 = cutlessCuts[node0][0].getDepth();
      auto levelNode1 = cutlessCuts[node1][0].getDepth();

      int level = std::max(levelNode0, levelNode1);
      if (level == 0)
        level = 1;

      std::set <std::string> leaves0;
      std::set <std::string> leaves1;
      if (levelNode0 == levelNode1){
        leaves0 = cutlessCuts[node0][0].getLeaves();
        leaves1 = cutlessCuts[node1][0].getLeaves();
      }
      else if (levelNode0 > levelNode1){
        leaves0 = cutlessCuts[node0][0].getLeaves();
        leaves1.insert(node1);
      }
      else if (levelNode0 < levelNode1){
        leaves0.insert(node0);
        leaves1 = cutlessCuts[node1][0].getLeaves();
      }

      std::set<std::string> merged = leaves0;
      merged.insert(leaves1.begin(), leaves1.end());

      if (merged.size() > lutSize){
        merged.clear();
        merged.insert(node0);
        merged.insert(node1);
        level += 1;
      }
      cutlessCuts[node].emplace_back(node, merged, level);
    }
  }
  return cutlessCuts;
}

std::unordered_map<std::string, std::vector<Cut>> Cuts::cutless(){
    int n = 0;
    std::set<std::string> currentWavyLine = blif.getPrimaryInputs();
    std::set<std::string> nextWavyLine;
    std::unordered_map<std::string, std::vector<Cut>> cutlessCuts;

    while (true) {
        nextWavyLine = blif.findNodesWithLimitedWavyInputs(6, currentWavyLine);

        if (nextWavyLine.size() == currentWavyLine.size()) {
            break;
        }

        for (auto& node : nextWavyLine) {
            cutlessCuts[node].emplace_back(node, blif.findWavyInputsOfNode(node, currentWavyLine), n);
        }

        n++;
        //currentWavyLine = nextWavyLine;
        currentWavyLine.insert(nextWavyLine.begin(), nextWavyLine.end());
        llvm::errs() << n << " " << currentWavyLine.size() << "\n";
    }

    for (auto& [node, cuts] : cutlessCuts) {
        cuts.erase(std::unique(cuts.begin(), cuts.end(), [](const Cut& a, const Cut& b) {
            return a.getLeaves() == b.getLeaves();
        }), cuts.end());
    }
  
    return cutlessCuts;
}


// Usage in your main cut enumeration function
std::unordered_map<std::string, std::vector<Cut>> Cuts::computeAllCuts() {

  //cuts = cutless();
  std::unordered_map<std::string, std::vector<Cut>> cutResults;

  for (const auto &node : blif.getNodesInOrder()) {

    if (blif.isPrimaryInput(node)) {
      cutResults[node].emplace_back(node, node, 0);
      continue;
    }
    
    auto fanins = blif.getFanins(node);

    if (fanins.size() == 1) {
      for (auto& cuts : cutResults[*fanins.begin()]){
        cutResults[node].emplace_back(node, cuts.getLeaves(), cuts.getDepth());
      }
      continue;
    }

    if (fanins.size() == 2) {
      std::set<std::string> insOfFanin1 = blif.getFanins(*fanins.begin());
      std::set<std::string> insOfFanin2 =
          blif.getFanins(*std::next(fanins.begin(), 1));

      if ((insOfFanin1 == insOfFanin2) && (!insOfFanin1.empty())) {
        for (auto& cuts : cutResults[*fanins.begin()]){
          cutResults[node].emplace_back(node, cuts.getLeaves(), cuts.getDepth());
        }

        int maxDepth = 0;
        for (const auto &cut : cutResults[*fanins.begin()]) {
          if (cut.getDepth() > maxDepth) {
            maxDepth = cut.getDepth();
          }
        }
        cutResults[node].emplace_back(node, node, maxDepth + 1);
        continue;
      }
    } 

    std::vector<std::vector<Cut>> faninCuts;
    for (const auto &fanin : fanins) {
      faninCuts.push_back(cutResults[fanin]);
    }

    std::vector<Cut> cutsEnumerated = enumerateCuts(node, faninCuts, lutSize);
    if (cutsEnumerated.size() == 1 && cutsEnumerated[0].getLeaves().empty()) {
      continue;
    }

    cutResults[node].insert(cutResults[node].end(), cutsEnumerated.begin(), cutsEnumerated.end());   
  }

  for (auto &[node, cuts] : cutResults) {

    std::sort(cuts.begin(), cuts.end(),
          [](const Cut &a, const Cut &b) {
            if (a.getDepth() == b.getDepth()) {
              return a.getLeaves().size() < b.getLeaves().size();
            }
            return a.getDepth() < b.getDepth();
          });

    cuts.erase(std::remove_if(cuts.begin(), cuts.end(),
                              [this](const Cut &cut) {
                                return cut.getLeaves().size() > lutSize;
                              }),
               cuts.end());

    cuts.erase(std::unique(cuts.begin(), cuts.end(),
                               [](const Cut &a, const Cut &b) {
                                 return a.getLeaves() == b.getLeaves();
                               }),
                   cuts.end());

  }

  return cutResults;  
}

void Cuts::runCutAlgos(bool computeAllCutsBool, bool cutlessBool, bool cutlessRealBool, bool anchors){

  std::unordered_map<std::string, std::vector<Cut>> tempCuts;

  if (computeAllCutsBool){
    auto cutsTemp1 = computeAllCuts();
    for (const auto &pair : cutsTemp1) {
      const std::string &node = pair.first;
      const std::vector<Cut> &cutVector = pair.second;
      tempCuts[node].insert(tempCuts[node].end(), cutVector.begin(), cutVector.end());
    }
  }
  if (cutlessBool){
    auto cutsTemp1 = cutless();
    for (const auto &pair : cutsTemp1) {
      const std::string &node = pair.first;
      const std::vector<Cut> &cutVector = pair.second;
      tempCuts[node].insert(tempCuts[node].end(), cutVector.begin(), cutVector.end());
    }
  }
  if (cutlessRealBool){
    auto cutsTemp1 = cutlessReal();
    for (const auto &pair : cutsTemp1) {
      const std::string &node = pair.first;
      const std::vector<Cut> &cutVector = pair.second;
      tempCuts[node].insert(tempCuts[node].end(), cutVector.begin(), cutVector.end());
    }
  }

  if (!anchors){
    for (const auto &pair : tempCuts) {
      const std::string &node = pair.first;
      const std::vector<Cut> &cutVector = pair.second;
      cuts[node].insert(cuts[node].end(), cutVector.begin(), cutVector.end());
    }
  }
  else{
    for (auto &pair : tempCuts) {
      std::vector<std::string> namesToRemove = {"__data_anchors_out", "__data_anchors_in", "__anchors_out", "__anchors_in"};
      std::function<std::string(std::string&)> removeAnchorsUtil = [namesToRemove](std::string& node) -> std::string {
        for (auto& nameToRemove : namesToRemove){
          size_t pos = node.find(nameToRemove);
          if (pos != std::string::npos) {
            node.erase(pos, std::string(nameToRemove).length());
            break;
          }
        }
        return node;
      };

      std::string node = pair.first;
      std::vector<Cut> &cutVector = pair.second;
      removeAnchorsUtil(node);

      for (auto &cut : cutVector) {
        std::string &root = cut.root;
        removeAnchorsUtil(root);
        
        std::set<std::string> leaves = cut.leaves;
        std::set<std::string> newLeaves;

        for (auto leaf : leaves) {
          std::string newLeaf = removeAnchorsUtil(leaf);
          newLeaves.insert(leaf);
        }

        cut.setLeaves(newLeaves);
      }

      cuts[node].insert(cuts[node].end(), cutVector.begin(), cutVector.end());
    }
  }

  for (auto &[node, cutVector] : cuts) {
    std::sort(cutVector.begin(), cutVector.end(),
          [](const Cut &a, const Cut &b) {
          if (a.getLeaves().size() == b.getLeaves().size()) {
            return a.getLeaves() < b.getLeaves();
          }
          return a.getLeaves().size() < b.getLeaves().size();
          });

    cutVector.erase(std::unique(cutVector.begin(), cutVector.end(),
                                [](const Cut &a, const Cut &b) {
                                  return a.getLeaves() == b.getLeaves();
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

  for (const auto &nodeCuts : cuts) {
    const std::string &node = nodeCuts.first;
    const std::vector<Cut> &cutList = nodeCuts.second;
    std::size_t numCuts = cutList.size();

    outFile << node << " " << numCuts << "\n";

    for (size_t i = 0; i < cutList.size(); ++i) {
      const Cut &cut = cutList[i];
      const std::set<std::string> &leaves = cut.getLeaves();
      std::size_t cutSize = leaves.size();
      outFile << "Cut #" << i << ": " << cutSize << " depth: " << cut.getDepth()
              << "\n";

      for (const auto &leaf : leaves) {
        outFile << '\t' << leaf << "\n";
      }
    }
  }
  outFile.close();
}

// CutEnumeration end
