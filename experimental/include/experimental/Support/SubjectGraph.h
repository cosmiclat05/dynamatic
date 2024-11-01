//===- SubjectGraph.h - Exp. support for MAPBUF buffer placement -------*- C++
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

#ifndef EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H
#define EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H

#include "BlifReader.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/HandshakeToHW.h"
#include "dynamatic/Dialect/HW/HWOpInterfaces.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/HW/HWTypes.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <string>

#include <boost/functional/hash/extensions.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

namespace dynamatic {
namespace experimental {

struct ChannelSignals {
  std::vector<Node *> dataSignals;
  Node *validSignal;
  Node *readySignal;
};

class BaseSubjectGraph {
public:
  static inline DenseMap<Operation *, BaseSubjectGraph *> moduleMap;
  static inline DenseMap<BaseSubjectGraph *, Operation *> subjectGraphMap;

protected:
  Operation *op;
  std::vector<Operation *> inputModules;
  std::vector<Operation *> outputModules;
  DenseMap<Operation *, unsigned int> inputModuleToResNum;
  DenseMap<Operation *, unsigned int> outputModuleToResNum;

  std::vector<BaseSubjectGraph *> inputSubjectGraphs;
  std::vector<BaseSubjectGraph *> outputSubjectGraphs;
  DenseMap<BaseSubjectGraph *, unsigned int> inputSubjectGraphToResNum;
  DenseMap<BaseSubjectGraph *, unsigned int> outputSubjectGraphToResNum;

  std::string fullPath = "/home/oyasar/full_integration/blif_files/";
  std::string moduleType;
  std::string uniqueName;
  BlifData *blifData;

  void assignSignals(ChannelSignals &signals, Node *node,
                     const std::string &nodeName);

public:
  BaseSubjectGraph() = default;
  BaseSubjectGraph(Operation *op) : op(op) {
    moduleMap[op] = this;
    subjectGraphMap[this] = op;
    moduleType = op->getName().getStringRef();
    uniqueName = getUniqueName(op);

    for (Value operand : op->getOperands()) {
      if (Operation *definingOp = operand.getDefiningOp()) {
        unsigned portNumber = operand.cast<OpResult>().getResultNumber();
        inputModules.push_back(definingOp);
        inputModuleToResNum[definingOp] = portNumber;
        // llvm::outs() << "Operation: " << uniqueName
        //              << " Defining op: " << definingOp->getName()
        //              << " channel num: " << portNumber << "\n";
      }
    }

    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        unsigned portNumber = result.cast<OpResult>().getResultNumber();
        outputModules.push_back(user);
        outputModuleToResNum[user] = portNumber;
        // llvm::outs() << "Operation: " << uniqueName
        //              << " Result op: " << user->getName()
        //              << " channel num: " << portNumber << "\n";
      }
    }

    // Extract the part after the dot in handshake.modulename
    size_t dotPosition = moduleType.find('.');
    if (dotPosition != std::string::npos) {
      moduleType = moduleType.substr(dotPosition + 1);
    } else {
      assert(false && "operation unsupported");
    }
  }

  BaseSubjectGraph(Operation *before, Operation *after) {
    // Constructor for adding a buffer between two operations
    auto *beforeSubjectGraph = moduleMap[before];
    auto *afterSubjectGraph = moduleMap[after];
    inputModules.push_back(before);
    outputModules.push_back(after);
    inputSubjectGraphs.push_back(beforeSubjectGraph);
    outputSubjectGraphs.push_back(afterSubjectGraph);
    inputModuleToResNum[before] = 0;
  }

  void replaceOpsBySubjectGraph() {
    // add the input and output subject graphs
    for (auto *inputModule : inputModules) {
      auto *inputSubjectGraph = moduleMap[inputModule];
      inputSubjectGraphs.push_back(inputSubjectGraph);
      inputSubjectGraphToResNum[inputSubjectGraph] =
          inputModuleToResNum[inputModule];
    }

    for (auto *outputModule : outputModules) {
      auto *outputSubjectGraph = moduleMap[outputModule];
      outputSubjectGraphs.push_back(outputSubjectGraph);
      outputSubjectGraphToResNum[outputSubjectGraph] =
          outputModuleToResNum[outputModule];
    }
  }

  static unsigned int getChannelNumber(BaseSubjectGraph *first,
                                       BaseSubjectGraph *second) {
    return first->outputSubjectGraphToResNum[second];
  }

  static void changeOutput(BaseSubjectGraph *graph, BaseSubjectGraph *newOutput,
                           BaseSubjectGraph *oldOutput) {
    auto it = std::find(graph->outputSubjectGraphs.begin(),
                        graph->outputSubjectGraphs.end(), oldOutput);
    unsigned int channelNumber = 0;
    if (it != graph->outputSubjectGraphs.end()) {
      channelNumber = graph->outputSubjectGraphToResNum[oldOutput];
      auto index = std::distance(graph->outputSubjectGraphs.begin(), it);
      graph->outputSubjectGraphs.erase(it);
      graph->outputSubjectGraphs.insert(
          graph->outputSubjectGraphs.begin() + index, newOutput);
      graph->outputSubjectGraphToResNum.erase(oldOutput);
    } else {
      llvm::errs() << "Output not found\n";
    }
    graph->outputSubjectGraphToResNum[newOutput] = channelNumber;
  }

  static void changeInput(BaseSubjectGraph *graph, BaseSubjectGraph *newInput,
                          BaseSubjectGraph *oldInput) {
    unsigned int channelNumber = 0;
    auto it = std::find(graph->inputSubjectGraphs.begin(),
                        graph->inputSubjectGraphs.end(), oldInput);
    if (it != graph->inputSubjectGraphs.end()) {
      auto index = std::distance(graph->inputSubjectGraphs.begin(), it);
      graph->inputSubjectGraphs.erase(it);
      graph->inputSubjectGraphs.insert(
          graph->inputSubjectGraphs.begin() + index, newInput);
      channelNumber = graph->inputSubjectGraphToResNum[oldInput];
      graph->inputSubjectGraphToResNum.erase(oldInput);
    } else {
      llvm::errs() << "Input not found\n";
    }
    graph->inputSubjectGraphToResNum[newInput] = channelNumber;
  }

  void appendVarsToPath(std::initializer_list<unsigned int> inputs) {
    fullPath += moduleType + "/";
    for (int input : inputs) {
      fullPath += std::to_string(input) + "/";
    }
    fullPath += moduleType + ".blif";
  }

  void connectSignals(Node *currentSignal, Node *beforeSignal) {
    beforeSignal->addFanout(currentSignal->getFanouts());
    currentSignal->setInput(false);
    beforeSignal->setOutput(false);
    beforeSignal->setInput(false);
    currentSignal->setOutput(false);
    beforeSignal->setChannelEdge(true);
    currentSignal->setChannelEdge(true);

    if (beforeSignal->isBlackboxOutputNode()) {
      beforeSignal->setInput(true);
    }

    for (auto &fanout : currentSignal->getFanouts()) {
      fanout->getFanins().erase(currentSignal);
      fanout->getFanins().insert(beforeSignal);
    }

    if (beforeSignal->getName().find("ready") != std::string::npos) {
      beforeSignal->setName(currentSignal->getName());
    }

    currentSignal = beforeSignal;
  }

  BlifData *getBlifData() const { return blifData; }

  static BaseSubjectGraph *getSubjectGraph(Operation *op) {
    return moduleMap[op];
  }

  std::string &getUniqueNameGraph() { return uniqueName; }

  std::string &getModuleType() { return moduleType; };

  std::vector<BaseSubjectGraph *> getInputSubjectGraphs() {
    return inputSubjectGraphs;
  }

  std::vector<BaseSubjectGraph *> getOutputSubjectGraphs() {
    return outputSubjectGraphs;
  }

  virtual ~BaseSubjectGraph() = default;
  virtual void connectInputNodes() = 0;
  virtual ChannelSignals &returnOutputNodes(unsigned int) = 0;
};

class ArithSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  std::unordered_map<unsigned int, ChannelSignals> inputNodes;
  ChannelSignals outputSignals;
  bool isBlackbox = false;

public:
  ArithSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::AddIOp, handshake::AndIOp, handshake::OrIOp,
              handshake::ShLIOp, handshake::ShRSIOp, handshake::ShRUIOp,
              handshake::SubIOp, handshake::XOrIOp, handshake::MulIOp,
              handshake::DivSIOp, handshake::DivUIOp>([&](auto) {
          // Bitwidth
          dataWidth =
              handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
          appendVarsToPath({dataWidth});
          if ((dataWidth > 4) && (llvm::isa<handshake::AddIOp>(op) ||
                                  llvm::isa<handshake::SubIOp>(op) ||
                                  llvm::isa<handshake::MulIOp>(op) ||
                                  llvm::isa<handshake::DivSIOp>(op) ||
                                  llvm::isa<handshake::DivUIOp>(op))) {
            isBlackbox = true;
          }
        })
        .Case<handshake::AddFOp, handshake::DivFOp, handshake::MaximumFOp,
              handshake::MinimumFOp, handshake::MulFOp, handshake::NegFOp,
              handshake::NotOp, handshake::SubFOp, handshake::SIToFPOp,
              handshake::FPToSIOp, handshake::AbsFOp, handshake::CmpFOp>(
            [&](auto) {
              assert(false && "Float not supported");
              return;
            })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("result") != std::string::npos) {
        assignSignals(outputSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
        if (isBlackbox && (nodeName.find("valid") == std::string::npos &&
                           nodeName.find("ready") == std::string::npos)) {
          node->setBlackboxOutput(true);
        }
      } else if (nodeName.find("lhs") != std::string::npos) {
        assignSignals(inputNodes[0], node, nodeName);
      } else if (nodeName.find("rhs") != std::string::npos) {
        assignSignals(inputNodes[1], node, nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  void connectInputNodes() override {
    for (unsigned int i = 0; i < inputNodes.size(); i++) {
      auto &currentInputNodes = inputNodes[i];
      auto *moduleBeforeSubjectGraph = inputSubjectGraphs[i];
      ChannelSignals &moduleBeforeOutputNodes =
          moduleBeforeSubjectGraph->returnOutputNodes(
              inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

      connectSignals(moduleBeforeOutputNodes.readySignal,
                     currentInputNodes.readySignal);
      connectSignals(currentInputNodes.validSignal,
                     moduleBeforeOutputNodes.validSignal);

      if (isBlackbox) {
        // the input to the blackbox should stay as an output signal
        for (auto *node : currentInputNodes.dataSignals) {
          node->setInput(false);
          node->setOutput(false);
        }
        continue;
      }

      for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
        connectSignals(currentInputNodes.dataSignals[j],
                       moduleBeforeOutputNodes.dataSignals[j]);
      }
    }
  }

  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override {
    return outputSignals;
  };
};

class CmpISubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  bool isBlackbox = false;
  std::unordered_map<unsigned int, ChannelSignals> inputNodes;
  ChannelSignals outputSignals;

public:
  CmpISubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::CmpIOp>([&](handshake::CmpIOp cmpIOp) {
          // Bitwidth
          dataWidth = handshake::getHandshakeTypeBitWidth(
              cmpIOp.getOperand(0).getType());
          appendVarsToPath({dataWidth});
          if (dataWidth > 4) {
            isBlackbox = true;
          }
        })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("result") != std::string::npos) {
        assignSignals(outputSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
        if (isBlackbox && (nodeName.find("valid") == std::string::npos &&
                           nodeName.find("ready") == std::string::npos)) {
          node->setBlackboxOutput(true);
        }
      } else if (nodeName.find("lhs") != std::string::npos) {
        assignSignals(inputNodes[0], node, nodeName);
      } else if (nodeName.find("rhs") != std::string::npos) {
        assignSignals(inputNodes[1], node, nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  void connectInputNodes() override {
    for (unsigned int i = 0; i < inputNodes.size(); i++) {
      auto &currentInputNodes = inputNodes[i];
      auto *moduleBeforeSubjectGraph = inputSubjectGraphs[i];
      ChannelSignals &moduleBeforeOutputNodes =
          moduleBeforeSubjectGraph->returnOutputNodes(
              inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

      connectSignals(moduleBeforeOutputNodes.readySignal,
                     currentInputNodes.readySignal);
      connectSignals(currentInputNodes.validSignal,
                     moduleBeforeOutputNodes.validSignal);

      if (isBlackbox) {
        // the input to the blackbox should stay as an output signal
        for (auto *node : currentInputNodes.dataSignals) {
          node->setInput(false);
          node->setOutput(false);
        }
        continue;
      }

      for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
        connectSignals(currentInputNodes.dataSignals[j],
                       moduleBeforeOutputNodes.dataSignals[j]);
      }
    }
  }

  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override {
    return outputSignals;
  };
};

class ForkSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int size = 0;
  unsigned int dataWidth = 0;
  std::unordered_map<unsigned int, ChannelSignals> outputNodes;
  ChannelSignals inputNodes;

public:
  ForkSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::ForkOp>([&](auto) {
          // Number of output channels and bitwidth
          size = op->getNumResults();
          dataWidth =
              handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());

          if (dataWidth == 0) {
            moduleType += "_dataless";
            appendVarsToPath({size});
          } else {
            moduleType += "_type";
            appendVarsToPath({size, dataWidth});
          }
        })

        .Case<handshake::LazyForkOp>([&](auto) {
          size = op->getNumResults();
          dataWidth =
              handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());

          if (dataWidth == 0) {
            moduleType += "_dataless";
            appendVarsToPath({size});
          } else {
            appendVarsToPath({size, dataWidth});
          }
        })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    auto generateNewNameRV = [](const std::string &nodeName)
        -> std::pair<std::string, unsigned int> {
      std::string newName = nodeName;
      std::string number;
      size_t bracketPos = newName.find('[');
      if (bracketPos != std::string::npos) {
        number = newName.substr(bracketPos + 1);
        number = number.substr(0, number.find_first_not_of("0123456789"));
      }
      size_t readyPos = newName.find("ready");
      size_t validPos = newName.find("valid");
      if (readyPos != std::string::npos) {
        newName = newName.substr(0, readyPos) + number + "_ready";
      } else if (validPos != std::string::npos) {
        newName = newName.substr(0, validPos) + number + "_valid";
      }
      return {newName, std::stoi(number)};
    };

    auto generateNewNameData = [&](const std::string &nodeName)
        -> std::pair<std::string, unsigned int> {
      std::string newName = nodeName;
      std::string number;
      size_t bracketPos = newName.find('[');
      if (bracketPos != std::string::npos) {
        number = newName.substr(bracketPos + 1);
        number = number.substr(0, number.find_first_not_of("0123456789"));
        newName = newName.substr(0, bracketPos);
      }
      unsigned int num = std::stoi(number);
      unsigned int newNumber = num / dataWidth;
      unsigned int remainder = (num % dataWidth);
      return {newName + "_" + std::to_string(newNumber) + "[" +
                  std::to_string(remainder) + "]",
              newNumber};
    };

    auto generateNewName = [&](const std::string &nodeName)
        -> std::pair<std::string, unsigned int> {
      if (nodeName.find("ready") != std::string::npos ||
          nodeName.find("valid") != std::string::npos) {
        return generateNewNameRV(nodeName);
      } else {
        return generateNewNameData(nodeName);
      }
    };

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("outs") != std::string::npos) {
        auto [newName, num] = generateNewName(nodeName);
        assignSignals(outputNodes[num], node, newName);
        node->setName(uniqueName + "_" + newName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      } else if (nodeName.find("ins") != std::string::npos &&
                 (node->isInput() || node->isOutput())) {
        assignSignals(inputNodes, node, nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  void connectInputNodes() override {
    auto &currentInputNodes = inputNodes;
    if (inputModules.empty()) {
      // start node
      return;
    }
    auto *moduleBeforeSubjectGraph = inputSubjectGraphs[0];
    ChannelSignals &moduleBeforeOutputNodes =
        moduleBeforeSubjectGraph->returnOutputNodes(
            inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

    connectSignals(moduleBeforeOutputNodes.readySignal,
                   currentInputNodes.readySignal);
    connectSignals(currentInputNodes.validSignal,
                   moduleBeforeOutputNodes.validSignal);

    for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
      connectSignals(currentInputNodes.dataSignals[j],
                     moduleBeforeOutputNodes.dataSignals[j]);
    }
  }

  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override {
    return outputNodes[channelIndex];
  };
};

class MuxSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int size = 0;
  unsigned int dataWidth = 0;
  unsigned int selectType = 0;
  std::unordered_map<unsigned int, ChannelSignals> inputNodes;
  ChannelSignals indexNodes;
  ChannelSignals outputNodes;

public:
  MuxSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
          // Number of input data channels, data bitwidth, and select bitwidth
          size = muxOp.getDataOperands().size();
          dataWidth =
              handshake::getHandshakeTypeBitWidth(muxOp.getResult().getType());
          selectType = handshake::getHandshakeTypeBitWidth(
              muxOp.getSelectOperand().getType());
          appendVarsToPath({size, dataWidth});
        })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("ins") != std::string::npos &&
          (node->isInput() || node->isOutput())) {
        size_t bracketPos = nodeName.find('[');
        std::string number = nodeName.substr(bracketPos + 1);
        number = number.substr(0, number.find_first_not_of("0123456789"));
        unsigned int num = std::stoi(number);
        if (nodeName.find("ready") == std::string::npos &&
            nodeName.find("valid") == std::string::npos) {
          num = num / dataWidth;
        }
        assignSignals(inputNodes[num], node, nodeName);
      } else if (nodeName.find("index") != std::string::npos) {
        assignSignals(indexNodes, node, nodeName);
      } else if (nodeName.find("outs") != std::string::npos) {
        assignSignals(outputNodes, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  void connectInputNodes() override {
    for (unsigned int i = 0; i < inputNodes.size(); i++) {
      auto &currentInputNodes = inputNodes[i];
      auto *moduleBeforeSubjectGraph = inputSubjectGraphs[i + 1];
      ChannelSignals &moduleBeforeOutputNodes =
          moduleBeforeSubjectGraph->returnOutputNodes(
              inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

      connectSignals(moduleBeforeOutputNodes.readySignal,
                     currentInputNodes.readySignal);
      connectSignals(currentInputNodes.validSignal,
                     moduleBeforeOutputNodes.validSignal);

      for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
        connectSignals(currentInputNodes.dataSignals[j],
                       moduleBeforeOutputNodes.dataSignals[j]);
      }
    }

    // index is the first input
    auto &currentIndexNodes = indexNodes;
    auto *moduleBeforeSubjectGraph = inputSubjectGraphs.front();
    ChannelSignals &moduleBeforeOutputNodes =
        moduleBeforeSubjectGraph->returnOutputNodes(
            inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

    connectSignals(moduleBeforeOutputNodes.readySignal,
                   currentIndexNodes.readySignal);
    connectSignals(currentIndexNodes.validSignal,
                   moduleBeforeOutputNodes.validSignal);

    for (unsigned int j = 0; j < currentIndexNodes.dataSignals.size(); j++) {

      connectSignals(currentIndexNodes.dataSignals[j],
                     moduleBeforeOutputNodes.dataSignals[j]);
    }
  }

  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override {
    return outputNodes;
  };
};

class ControlMergeSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int size = 0;
  unsigned int dataWidth = 0;
  unsigned int indexType = 0;
  std::unordered_map<unsigned int, ChannelSignals> inputNodes;
  ChannelSignals indexNodes;
  ChannelSignals outputNodes;

public:
  ControlMergeSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::ControlMergeOp>(
            [&](handshake::ControlMergeOp cmergeOp) {
              // Number of input data channels, data bitwidth, and index
              // bitwidth
              size = cmergeOp.getDataOperands().size();
              dataWidth = handshake::getHandshakeTypeBitWidth(
                  cmergeOp.getResult().getType());
              indexType = handshake::getHandshakeTypeBitWidth(
                  cmergeOp.getIndex().getType());
              if (dataWidth == 0) {
                moduleType += "_dataless";
                appendVarsToPath({size, indexType});
              } else {
                assert(false && "ControlMerge with data width not supported");
                // appendVarsToPath({size, dataWidth, indexType});
              }
            })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("ins") != std::string::npos &&
          (node->isInput() || node->isOutput())) {
        if (size == 1) {
          assignSignals(inputNodes[0], node, nodeName);
          continue;
        }
        size_t bracketPos = nodeName.find('[');
        std::string number = nodeName.substr(bracketPos + 1);
        number = number.substr(0, number.find_first_not_of("0123456789"));
        unsigned int num = std::stoi(number);
        if (nodeName.find("ready") == std::string::npos &&
            nodeName.find("valid") == std::string::npos) {
          num = num / dataWidth;
        }
        assignSignals(inputNodes[num], node, nodeName);
      } else if (nodeName.find("index") != std::string::npos) {
        assignSignals(indexNodes, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find("outs") != std::string::npos) {
        assignSignals(outputNodes, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  };

  void connectInputNodes() override {
    for (unsigned int i = 0; i < inputNodes.size(); i++) {
      auto &currentInputNodes = inputNodes[i];
      auto *moduleBeforeSubjectGraph = inputSubjectGraphs[i];
      ChannelSignals &moduleBeforeOutputNodes =
          moduleBeforeSubjectGraph->returnOutputNodes(
              inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

      connectSignals(moduleBeforeOutputNodes.readySignal,
                     currentInputNodes.readySignal);
      connectSignals(currentInputNodes.validSignal,
                     moduleBeforeOutputNodes.validSignal);

      for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
        connectSignals(currentInputNodes.dataSignals[j],
                       moduleBeforeOutputNodes.dataSignals[j]);
      }
    }
  };

  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override {
    return (channelIndex == 0) ? outputNodes : indexNodes;
  };
};

class ConditionalBranchSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  ChannelSignals conditionNodes;
  ChannelSignals inputNodes;
  std::unordered_map<unsigned int, ChannelSignals> outputNodes;

public:
  ConditionalBranchSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::ConditionalBranchOp>(
            [&](handshake::ConditionalBranchOp cbrOp) {
              dataWidth = handshake::getHandshakeTypeBitWidth(
                  cbrOp.getDataOperand().getType());
              if (dataWidth == 0) {
                moduleType += "_dataless";
                appendVarsToPath({});
              } else {
                appendVarsToPath({dataWidth});
              }
            })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("true") != std::string::npos) {
        assignSignals(outputNodes[0], node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find("false") != std::string::npos) {
        assignSignals(outputNodes[1], node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find("condition") != std::string::npos) {
        assignSignals(conditionNodes, node, nodeName);
      } else if (nodeName.find("data") != std::string::npos) {
        assignSignals(inputNodes, node, nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  void connectInputNodes() override {
    {
      auto &currentInputNodes = conditionNodes;
      auto *moduleBeforeSubjectGraph = inputSubjectGraphs[0];
      ChannelSignals &moduleBeforeOutputNodes =
          moduleBeforeSubjectGraph->returnOutputNodes(
              inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

      connectSignals(moduleBeforeOutputNodes.readySignal,
                     currentInputNodes.readySignal);
      connectSignals(currentInputNodes.validSignal,
                     moduleBeforeOutputNodes.validSignal);

      for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
        connectSignals(currentInputNodes.dataSignals[j],
                       moduleBeforeOutputNodes.dataSignals[j]);
      }
    }
    {
      auto &currentInputNodes = inputNodes;
      auto *moduleBeforeSubjectGraph = inputSubjectGraphs[1];
      ChannelSignals &moduleBeforeOutputNodes =
          moduleBeforeSubjectGraph->returnOutputNodes(
              inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);
      connectSignals(moduleBeforeOutputNodes.readySignal,
                     currentInputNodes.readySignal);
      connectSignals(currentInputNodes.validSignal,
                     moduleBeforeOutputNodes.validSignal);

      for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
        connectSignals(currentInputNodes.dataSignals[j],
                       moduleBeforeOutputNodes.dataSignals[j]);
      }
    }
  }

  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override {
    return outputNodes[channelIndex];
  }
};

class SourceSubjectGraph : public BaseSubjectGraph {
  ChannelSignals outputSignals;

public:
  SourceSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::SourceOp>([&](auto) {
          // No discriminating parameters
          appendVarsToPath({});
        })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("outs") != std::string::npos) {
        assignSignals(outputSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  void connectInputNodes() override {
    // No input nodes
  }

  ChannelSignals &returnOutputNodes(unsigned int) override {
    return outputSignals;
  }
};

class LoadSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  unsigned int addrType = 0;
  ChannelSignals addrInSignals;
  ChannelSignals addrOutSignals;
  ChannelSignals dataOutSignals;

public:
  LoadSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::LoadOpInterface>(
            [&](handshake::LoadOpInterface loadOp) {
              dataWidth = handshake::getHandshakeTypeBitWidth(
                  loadOp.getDataInput().getType());
              addrType = handshake::getHandshakeTypeBitWidth(
                  loadOp.getAddressInput().getType());
              appendVarsToPath({addrType, dataWidth});
            })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("addrIn") != std::string::npos) {
        assignSignals(addrInSignals, node, nodeName);
      } else if (nodeName.find("addrOut") != std::string::npos) {
        assignSignals(addrOutSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find("dataOut") != std::string::npos) {
        assignSignals(dataOutSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  void connectInputNodes() override {
    auto &currentInputNodes = addrInSignals;
    auto *moduleBeforeSubjectGraph = inputSubjectGraphs[0];
    ChannelSignals &moduleBeforeOutputNodes =
        moduleBeforeSubjectGraph->returnOutputNodes(
            inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

    connectSignals(moduleBeforeOutputNodes.readySignal,
                   currentInputNodes.readySignal);
    connectSignals(currentInputNodes.validSignal,
                   moduleBeforeOutputNodes.validSignal);

    for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
      connectSignals(currentInputNodes.dataSignals[j],
                     moduleBeforeOutputNodes.dataSignals[j]);
    }
  }

  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override {
    return (channelIndex == 0) ? addrOutSignals : dataOutSignals;
  }
};

class StoreSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  unsigned int addrType = 0;
  ChannelSignals dataInSignals;
  ChannelSignals addrInSignals;
  ChannelSignals addrOutSignals;

public:
  StoreSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::StoreOpInterface>(
            [&](handshake::StoreOpInterface storeOp) {
              dataWidth = handshake::getHandshakeTypeBitWidth(
                  storeOp.getDataInput().getType());
              addrType = handshake::getHandshakeTypeBitWidth(
                  storeOp.getAddressInput().getType());
              appendVarsToPath({addrType, dataWidth});
            })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("dataIn") != std::string::npos) {
        assignSignals(dataInSignals, node, nodeName);
      } else if (nodeName.find("addrIn") != std::string::npos) {
        assignSignals(addrInSignals, node, nodeName);
      } else if (nodeName.find("addrOut") != std::string::npos) {
        assignSignals(addrOutSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  void connectInputNodes() override {
    auto &currentInputNodes = addrInSignals;
    auto *moduleBeforeSubjectGraph = inputSubjectGraphs[0];
    ChannelSignals &moduleBeforeOutputNodes =
        moduleBeforeSubjectGraph->returnOutputNodes(
            inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

    connectSignals(moduleBeforeOutputNodes.readySignal,
                   currentInputNodes.readySignal);
    connectSignals(currentInputNodes.validSignal,
                   moduleBeforeOutputNodes.validSignal);

    for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
      connectSignals(currentInputNodes.dataSignals[j],
                     moduleBeforeOutputNodes.dataSignals[j]);
    }

    auto &currentInputNodes2 = dataInSignals;
    auto *moduleBeforeSubjectGraph2 = inputSubjectGraphs[1];
    ChannelSignals &moduleBeforeOutputNodes2 =
        moduleBeforeSubjectGraph2->returnOutputNodes(
            inputSubjectGraphToResNum[moduleBeforeSubjectGraph2]);

    connectSignals(moduleBeforeOutputNodes2.readySignal,
                   currentInputNodes2.readySignal);
    connectSignals(currentInputNodes2.validSignal,
                   moduleBeforeOutputNodes2.validSignal);

    for (unsigned int j = 0; j < currentInputNodes2.dataSignals.size(); j++) {
      connectSignals(currentInputNodes2.dataSignals[j],
                     moduleBeforeOutputNodes2.dataSignals[j]);
    }
  }

  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override {
    return addrOutSignals;
  }
};

class ConstantSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  // uint64_t constantValue = 0;
  ChannelSignals controlSignals;
  ChannelSignals outputSignals;

public:
  ConstantSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
          handshake::ChannelType cstType = cstOp.getResult().getType();
          unsigned bitwidth = cstType.getDataBitWidth();
          dataWidth = bitwidth;
          appendVarsToPath({dataWidth});

          if (bitwidth > 64) {
            cstOp.emitError() << "Constant value has bitwidth " << bitwidth
                              << ", but we only support up to 64.";
            return;
          }
        })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("outs") != std::string::npos) {
        node->setName(uniqueName + "_" + nodeName);
        assignSignals(outputSignals, node, nodeName);
      } else if (nodeName.find("ctrl") != std::string::npos) {
        assignSignals(controlSignals, node, nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  void connectInputNodes() override {
    auto &currentControlSignals = controlSignals;
    auto *moduleBeforeSubjectGraph = inputSubjectGraphs[0];
    ChannelSignals &moduleBeforeOutputNodes =
        moduleBeforeSubjectGraph->returnOutputNodes(
            inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

    connectSignals(moduleBeforeOutputNodes.readySignal,
                   currentControlSignals.readySignal);
    connectSignals(currentControlSignals.validSignal,
                   moduleBeforeOutputNodes.validSignal);

    for (unsigned int j = 0; j < currentControlSignals.dataSignals.size();
         j++) {
      connectSignals(currentControlSignals.dataSignals[j],
                     moduleBeforeOutputNodes.dataSignals[j]);
    }
  }

  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override {
    return outputSignals;
  }
};

class ExtTruncSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int inputWidth = 0;
  unsigned int outputWidth = 0;
  ChannelSignals inputSignals;
  ChannelSignals outputSignals;

public:
  ExtTruncSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::ExtFOp,
              handshake::TruncIOp, handshake::TruncFOp>([&](auto extOp) {
          inputWidth =
              handshake::getHandshakeTypeBitWidth(extOp.getOperand().getType());
          outputWidth =
              handshake::getHandshakeTypeBitWidth(extOp.getResult().getType());
          appendVarsToPath({inputWidth, outputWidth});
        })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("ins") != std::string::npos &&
          (node->isInput() || node->isOutput())) {
        assignSignals(inputSignals, node, nodeName);
      } else if (nodeName.find("outs") != std::string::npos) {
        assignSignals(outputSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  };

  void connectInputNodes() override {
    auto &currentInputNodes = inputSignals;
    auto *moduleBeforeSubjectGraph = inputSubjectGraphs[0];
    ChannelSignals &moduleBeforeOutputNodes =
        moduleBeforeSubjectGraph->returnOutputNodes(
            inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

    connectSignals(moduleBeforeOutputNodes.readySignal,
                   currentInputNodes.readySignal);
    connectSignals(currentInputNodes.validSignal,
                   moduleBeforeOutputNodes.validSignal);

    for (unsigned int j = 0; j < inputWidth; j++) {
      connectSignals(currentInputNodes.dataSignals[j],
                     moduleBeforeOutputNodes.dataSignals[j]);
    }
  };

  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override {
    return outputSignals;
  };
};

class SelectSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  std::unordered_map<unsigned int, ChannelSignals> inputSignals;
  ChannelSignals outputSignals;

public:
  SelectSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::SelectOp>([&](auto selectOp) {
          dataWidth = handshake::getHandshakeTypeBitWidth(
              selectOp->getOperand(1).getType());
          appendVarsToPath({dataWidth});
        })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("trueValue") != std::string::npos &&
          (node->isInput() || node->isOutput())) {
        assignSignals(inputSignals[1], node, nodeName);
      } else if (nodeName.find("falseValue") != std::string::npos &&
                 (node->isInput() || node->isOutput())) {
        assignSignals(inputSignals[2], node, nodeName);
      } else if (nodeName.find("condition") != std::string::npos &&
                 (node->isInput() || node->isOutput())) {
        assignSignals(inputSignals[0], node, nodeName);
      } else if (nodeName.find("result") != std::string::npos) {
        assignSignals(outputSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  };

  void connectInputNodes() override {
    for (unsigned int i = 0; i < inputSignals.size(); i++) {
      auto &currentInputNodes = inputSignals[i];
      auto *moduleBeforeSubjectGraph = inputSubjectGraphs[i];
      ChannelSignals &moduleBeforeOutputNodes =
          moduleBeforeSubjectGraph->returnOutputNodes(
              inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

      connectSignals(moduleBeforeOutputNodes.readySignal,
                     currentInputNodes.readySignal);
      connectSignals(currentInputNodes.validSignal,
                     moduleBeforeOutputNodes.validSignal);

      for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
        connectSignals(currentInputNodes.dataSignals[j],
                       moduleBeforeOutputNodes.dataSignals[j]);
      }
    }
  }

  ChannelSignals &returnOutputNodes(unsigned int channelIndex) override {
    return outputSignals;
  };
};

class BranchSinkSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  ChannelSignals inputSignals;
  ChannelSignals outputSignals;

public:
  BranchSinkSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::BranchOp, handshake::SinkOp>([&](auto) {
          dataWidth =
              handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
          if (dataWidth == 0) {
            moduleType += "_dataless";
            appendVarsToPath({});
          } else {
            appendVarsToPath({dataWidth});
          }
        })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("ins") != std::string::npos &&
          (node->isInput() || node->isOutput())) {
        assignSignals(inputSignals, node, nodeName);
      } else if (nodeName.find("outs") != std::string::npos) {
        assignSignals(outputSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos ||
                 nodeName.find("dataReg") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  void connectInputNodes() override {
    if (inputModules.empty()) {
      // start node
      return;
    }
    auto &currentInputNodes = inputSignals;

    auto *moduleBeforeSubjectGraph = inputSubjectGraphs[0];
    ChannelSignals &moduleBeforeOutputNodes =
        moduleBeforeSubjectGraph->returnOutputNodes(
            inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

    connectSignals(moduleBeforeOutputNodes.readySignal,
                   currentInputNodes.readySignal);
    connectSignals(currentInputNodes.validSignal,
                   moduleBeforeOutputNodes.validSignal);

    for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
      connectSignals(currentInputNodes.dataSignals[j],
                     moduleBeforeOutputNodes.dataSignals[j]);
    }
  }

  ChannelSignals &returnOutputNodes(unsigned int) override {
    return outputSignals;
  }
};

class BufferSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  ChannelSignals inputSignals;
  ChannelSignals outputSignals;

public:
  BufferSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::BufferOp>([&](handshake::BufferOp bufferOp) {
          auto params =
              bufferOp->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
          auto optTiming =
              params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);

          if (auto timing =
                  dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
            handshake::TimingInfo info = timing.getInfo();
            if (info == handshake::TimingInfo::oehb())
              moduleType = "oehb";
            if (info == handshake::TimingInfo::tehb())
              moduleType = "tehb";
          }

          dataWidth =
              handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
          if (dataWidth == 0) {
            moduleType += "_dataless";
            appendVarsToPath({});
          } else {
            appendVarsToPath({dataWidth});
          }
        })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("ins") != std::string::npos &&
          (node->isInput() || node->isOutput())) {
        assignSignals(inputSignals, node, nodeName);
      } else if (nodeName.find("outs") != std::string::npos) {
        assignSignals(outputSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos ||
                 nodeName.find("dataReg") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  BufferSubjectGraph(Operation *op1, Operation *op2, std::string &bufferType)
      : BaseSubjectGraph() {
    subjectGraphMap[this] = nullptr;
    moduleType = bufferType;

    BaseSubjectGraph *graph1 = moduleMap[op1];
    BaseSubjectGraph *graph2 = moduleMap[op2];
    inputSubjectGraphs.push_back(graph1);
    outputSubjectGraphs.push_back(graph2);

    static unsigned int bufferCount;
    // find the channel number and datawidth of the channel
    unsigned int channelNum = getChannelNumber(graph1, graph2);
    outputSubjectGraphToResNum[graph2] = channelNum;
    inputSubjectGraphToResNum[graph1] = channelNum;
    // get the channel width
    ChannelSignals &channel = graph1->returnOutputNodes(channelNum);
    dataWidth = channel.dataSignals.size();
    // set the unique name
    uniqueName = "oehb_" + std::to_string(bufferCount++);

    changeOutput(graph1, this, graph2);
    changeInput(graph2, this, graph1);

    if (dataWidth == 0) {
      moduleType += "_dataless";
      appendVarsToPath({});
    } else {
      appendVarsToPath({dataWidth});
    }

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("ins") != std::string::npos &&
          (node->isInput() || node->isOutput())) {
        assignSignals(inputSignals, node, nodeName);
      } else if (nodeName.find("outs") != std::string::npos) {
        assignSignals(outputSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos ||
                 nodeName.find("dataReg") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  BufferSubjectGraph(BufferSubjectGraph *graph1, Operation *op2,
                     std::string &bufferType)
      : BaseSubjectGraph() {
    subjectGraphMap[this] = nullptr;
    moduleType = bufferType;

    BaseSubjectGraph *graph2 = moduleMap[op2];
    inputSubjectGraphs.push_back(graph1);
    outputSubjectGraphs.push_back(graph2);

    static unsigned int bufferCount;
    // find the channel number and datawidth of the channel
    unsigned int channelNum = getChannelNumber(graph1, graph2);
    outputSubjectGraphToResNum[graph2] = channelNum;
    inputSubjectGraphToResNum[graph1] = channelNum;
    // get the channel width
    ChannelSignals &channel = graph1->returnOutputNodes(channelNum);
    dataWidth = channel.dataSignals.size();
    // set the unique name
    uniqueName = "tehb_" + std::to_string(bufferCount++);

    changeOutput(graph1, this, graph2);
    changeInput(graph2, this, graph1);

    if (dataWidth == 0) {
      moduleType += "_dataless";
      appendVarsToPath({});
    } else {
      appendVarsToPath({dataWidth});
    }

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("ins") != std::string::npos &&
          (node->isInput() || node->isOutput())) {
        assignSignals(inputSignals, node, nodeName);
      } else if (nodeName.find("outs") != std::string::npos) {
        assignSignals(outputSignals, node, nodeName);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos ||
                 nodeName.find("dataReg") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  void connectInputNodes() override {
    auto &currentInputNodes = inputSignals;
    auto *moduleBeforeSubjectGraph = inputSubjectGraphs[0];
    ChannelSignals &moduleBeforeOutputNodes =
        moduleBeforeSubjectGraph->returnOutputNodes(
            inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

    connectSignals(moduleBeforeOutputNodes.readySignal,
                   currentInputNodes.readySignal);
    connectSignals(currentInputNodes.validSignal,
                   moduleBeforeOutputNodes.validSignal);

    for (unsigned int j = 0; j < currentInputNodes.dataSignals.size(); j++) {
      connectSignals(currentInputNodes.dataSignals[j],
                     moduleBeforeOutputNodes.dataSignals[j]);
    }
  }

  ChannelSignals &returnOutputNodes(unsigned int) override {
    return outputSignals;
  }
};

class OperationDifferentiator {
  Operation *op;

public:
  static inline DenseMap<Operation *, BaseSubjectGraph *> moduleMap;

  OperationDifferentiator(Operation *ops) : op(ops) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::InstanceOp>([&](handshake::InstanceOp instOp) {
          // op->emitRemark("Instance Op");
        })
        .Case<handshake::ForkOp, handshake::LazyForkOp>(
            [&](auto) { moduleMap[op] = new ForkSubjectGraph(op); })
        .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
          moduleMap[op] = new MuxSubjectGraph(op);
        })
        .Case<handshake::ControlMergeOp>(
            [&](handshake::ControlMergeOp cmergeOp) {
              moduleMap[op] = new ControlMergeSubjectGraph(op);
            })
        .Case<handshake::MergeOp>([&](auto) { op->emitRemark("Merge Op"); })
        .Case<handshake::JoinOp>([&](auto) { op->emitRemark("Join Op"); })
        .Case<handshake::BranchOp, handshake::SinkOp>(
            [&](auto) { moduleMap[op] = new BranchSinkSubjectGraph(op); })
        .Case<handshake::BufferOp, handshake::SinkOp>(
            [&](auto) { moduleMap[op] = new BufferSubjectGraph(op); })
        .Case<handshake::ConditionalBranchOp>(
            [&](handshake::ConditionalBranchOp cbrOp) {
              moduleMap[op] = new ConditionalBranchSubjectGraph(op);
            })
        .Case<handshake::SourceOp>([&](auto) {
          moduleMap[op] = new SourceSubjectGraph(op);
        })
        .Case<handshake::LoadOpInterface>(
            [&](handshake::LoadOpInterface loadOp) {
              moduleMap[op] = new LoadSubjectGraph(op);
            })
        .Case<handshake::StoreOpInterface>(
            [&](handshake::StoreOpInterface storeOp) {
              moduleMap[op] = new StoreSubjectGraph(op);
            })
        .Case<handshake::SharingWrapperOp>(
            [&](handshake::SharingWrapperOp sharingWrapperOp) {
              assert(false && "operation unsupported");
            })
        .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
          moduleMap[op] = new ConstantSubjectGraph(op);
        })
        .Case<handshake::AddFOp, handshake::DivFOp, handshake::MaximumFOp,
              handshake::MinimumFOp, handshake::MulFOp, handshake::NegFOp,
              handshake::NotOp, handshake::SubFOp, handshake::SIToFPOp,
              handshake::FPToSIOp, handshake::AbsFOp, handshake::CmpFOp>(
            [&](auto) {
              op->emitError() << "Float not supported";
              return;
            })
        .Case<handshake::AddIOp, handshake::AndIOp, handshake::OrIOp,
              handshake::ShLIOp, handshake::ShRSIOp, handshake::ShRUIOp,
              handshake::SubIOp, handshake::XOrIOp, handshake::MulIOp,
              handshake::DivSIOp, handshake::DivUIOp>(
            [&](auto) { moduleMap[op] = new ArithSubjectGraph(op); })
        .Case<handshake::SelectOp>([&](handshake::SelectOp selectOp) {
          moduleMap[op] = new SelectSubjectGraph(op);
        })
        .Case<handshake::CmpIOp>([&](handshake::CmpIOp cmpIOp) {
          moduleMap[op] = {moduleMap[op] = new CmpISubjectGraph(op)};
        })
        .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::ExtFOp,
              handshake::TruncIOp, handshake::TruncFOp>(
            [&](auto) { moduleMap[op] = new ExtTruncSubjectGraph(op); })
        .Default([&](auto) {
          llvm::errs() << "No subject graph can be generated for this "
                          "operation: "
                       << op->getName() << "\n";
          return;
        });
  }
};

class SubjectGraphGenerator {
public:
  SubjectGraphGenerator(handshake::FuncOp funcOp) {
    //create the subject graph for each operation
    funcOp.walk(
        [&](Operation *op) { (experimental::OperationDifferentiator(op)); });

    //initialize the input and output modules of each subject graph
    for (auto &module : experimental::OperationDifferentiator::moduleMap) {
      module.second->replaceOpsBySubjectGraph();
    }
  }
};
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H