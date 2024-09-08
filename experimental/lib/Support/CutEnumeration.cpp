//===-- CutEnumeration.cpp - Exp. support for MAPBUF buffer placement -----*- C++
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

#include <set>
#include <stack>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "llvm/Support/Path.h"

#include "experimental/Support/CutEnumeration.h"

using namespace dynamatic::experimental;

void Cuts::readFromFile(const std::string& filename) {
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
        if (line.empty()) continue;
        if (line[0] != '\t') {
            std::istringstream iss(line);
            iss >> node >> numberOfCuts;
            for (int i = 0; i < numberOfCuts; i++){
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
              }
              else
              {
                llvm::errs() << "No cut found!\n";
                llvm::errs() << line << "\n";
              }
            }
        } 
        else {
          llvm::errs() << "Tab found!\n";
          llvm::errs() << line << "\n";
        }
    }
}

// CutEnumeration end
