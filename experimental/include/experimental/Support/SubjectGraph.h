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

class BaseSubjectGraph {
protected:
  Operation *op;
  SmallVector<Operation *> inputs;
  SmallVector<Operation *> outputs;
  std::string fullPath = "/home/oyasar/full_integration/blif_files/";
  std::string moduleType;
  std::string uniqueName;
  BlifData *blifData;

public:
  BaseSubjectGraph(Operation *op) : op(op) {
    moduleType = op->getName().getStringRef();
    uniqueName = getUniqueName(op);

    // Extract the part after the dot in handshake.modulename
    size_t dotPosition = moduleType.find('.');
    if (dotPosition != std::string::npos) {
      moduleType = moduleType.substr(dotPosition + 1);
    } else {
      assert(false && "operation unsupported");
    }
  }

  void appendVarsToPath(std::initializer_list<unsigned int> inputs) {
    fullPath += moduleType + "/";
    for (int input : inputs) {
      fullPath += std::to_string(input) + "/";
    }
    fullPath += moduleType + ".blif";
  }
};

class ArithSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  std::vector<Node *> lhsNodes;
  std::vector<Node *> rhsNodes;
  std::vector<Node *> outputNodes;

public:
  ArithSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::AddIOp, handshake::AndIOp, handshake::OrIOp,
              handshake::ShLIOp, handshake::ShRSIOp, handshake::ShRUIOp,
              handshake::SubIOp, handshake::XOrIOp>([&](auto) {
          // Bitwidth
          dataWidth =
              handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
          appendVarsToPath({dataWidth});
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

    appendVarsToPath({dataWidth});
    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("result") != std::string::npos) {
        outputNodes.push_back(node);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }

      if (nodeName.find("lhs") != std::string::npos) {
        lhsNodes.push_back(node);
      } else if (nodeName.find("rhs") != std::string::npos) {
        rhsNodes.push_back(node);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }
};

class ForkSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int size = 0;
  unsigned int dataWidth = 0;
  std::unordered_map<unsigned int, std::vector<Node *>> channelNodes;

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
      unsigned int remainder = num % dataWidth;
      return {newName + "_" + std::to_string(newNumber) + "[" +
                  std::to_string(remainder) + "]",
              num};
    };

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);
    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("outs_valid") != std::string::npos) {
        node->setName(uniqueName + "_" + generateNewNameRV(nodeName).first);
        channelNodes[generateNewNameRV(nodeName).second].push_back(node);
      } else if (nodeName.find("outs_ready") != std::string::npos) {
        node->setName(uniqueName + "_" + generateNewNameRV(nodeName).first);
        channelNodes[generateNewNameRV(nodeName).second].push_back(node);
      } else if (nodeName.find("outs") != std::string::npos) {
        node->setName(uniqueName + "_" + generateNewNameData(nodeName).first);
        channelNodes[generateNewNameData(nodeName).second].push_back(node);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }
};

class MuxSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int size = 0;
  unsigned int dataWidth = 0;
  unsigned int selectType = 0;
  std::unordered_map<unsigned int, std::vector<Node *>> insNodes;
  std::vector<Node *> indexNodes;
  std::vector<Node *> outputNodes;

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
          appendVarsToPath({size, dataWidth, selectType});
        })
        .Default([&](auto) {
          assert(false && "Operation does not match any supported type");
          return;
        });

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("ins") != std::string::npos) {
        size_t bracketPos = nodeName.find('[');
        if (bracketPos != std::string::npos) {
          std::string number = nodeName.substr(bracketPos + 1);
          number = number.substr(0, number.find_first_not_of("0123456789"));
          unsigned int num = std::stoi(number);
          unsigned int newNumber = num / dataWidth;
          insNodes[newNumber].push_back(node);
        }
      } else if (nodeName.find("index") != std::string::npos) {
        indexNodes.push_back(node);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find("outs") != std::string::npos) {
        outputNodes.push_back(node);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }

  class ControlMergeSubjectGraph : public BaseSubjectGraph {
  private:
    unsigned int size = 0;
    unsigned int dataWidth = 0;
    unsigned int indexType = 0;
    std::unordered_map<unsigned int, std::vector<Node *>> insNodes;
    std::vector<Node *> indexNodes;
    std::vector<Node *> outputNodes;

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
        if (nodeName.find("ins") != std::string::npos) {
          size_t bracketPos = nodeName.find('[');
          if (bracketPos != std::string::npos) {
            std::string number = nodeName.substr(bracketPos + 1);
            number = number.substr(0, number.find_first_not_of("0123456789"));
            unsigned int num = std::stoi(number);
            unsigned int newNumber = num / dataWidth;
            insNodes[newNumber].push_back(node);
          }
        } else if (nodeName.find("index") != std::string::npos) {
          indexNodes.push_back(node);
          node->setName(uniqueName + "_" + nodeName);
        } else if (nodeName.find("outs") != std::string::npos) {
          outputNodes.push_back(node);
          node->setName(uniqueName + "_" + nodeName);
        } else if (nodeName.find(".") != std::string::npos) {
          node->setName(uniqueName + "." + nodeName);
        }
      }

      blifData->generateBlifFile(
          "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
          ".blif");
    }
  };
};

class ConditionalBranchSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  std::vector<Node *> trueBranchNodes;
  std::vector<Node *> falseBranchNodes;
  std::vector<Node *> conditionNodes;
  std::vector<Node *> dataNodes;

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
        trueBranchNodes.push_back(node);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find("false") != std::string::npos) {
        falseBranchNodes.push_back(node);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find("condition") != std::string::npos) {
        conditionNodes.push_back(node);
      } else if (nodeName.find("data") != std::string::npos) {
        dataNodes.push_back(node);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }
};

class SourceSubjectGraph : public BaseSubjectGraph {
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
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }
};

class LoadSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  unsigned int addrType = 0;
  std::vector<Node *> addrInNodes;
  std::vector<Node *> addrOutNodes;
  std::vector<Node *> dataOutNodes;

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
        addrInNodes.push_back(node);
      } else if (nodeName.find("addrOut") != std::string::npos) {
        node->setName(uniqueName + "_" + nodeName);
        addrOutNodes.push_back(node);
      } else if (nodeName.find("dataOut") != std::string::npos) {
        node->setName(uniqueName + "_" + nodeName);
        dataOutNodes.push_back(node);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }
};

class StoreSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  unsigned int addrType = 0;
  std::vector<Node *> dataInNodes;
  std::vector<Node *> addrInNodes;
  std::vector<Node *> addrOutNodes;

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
      dataInNodes.push_back(node);
      } else if (nodeName.find("addrIn") != std::string::npos) {
      node->setName(uniqueName + "_" + nodeName);
      addrInNodes.push_back(node);
      } else if (nodeName.find("addrOut") != std::string::npos) {
      node->setName(uniqueName + "_" + nodeName);
      addrOutNodes.push_back(node);
      } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }
};

class ConstantSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int dataWidth = 0;
  uint64_t constantValue = 0;
  std::vector<Node *> outsNodes;
  std::vector<Node*> controlNodes;

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
        outsNodes.push_back(node);
      } 
      else if (nodeName.find("control") != std::string::npos) {
        controlNodes.push_back(node);
      }
      else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }
};

class ExtTruncSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int inputWidth = 0;
  unsigned int outputWidth = 0;
  std::vector<Node *> inputNodes;
  std::vector<Node *> outputNodes;

public:
  ExtTruncSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::ExtFOp>(
            [&](auto extOp) {
              inputWidth = handshake::getHandshakeTypeBitWidth(extOp.getOperand().getType());
              outputWidth = handshake::getHandshakeTypeBitWidth(extOp.getResult().getType());
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
      if (nodeName.find("ins") != std::string::npos) {
        inputNodes.push_back(node);
      } else if (nodeName.find("outs") != std::string::npos) {
        outputNodes.push_back(node);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }
};

class TruncSubjectGraph : public BaseSubjectGraph {
private:
  unsigned int inputWidth = 0;
  unsigned int outputWidth = 0;
  std::vector<Node *> inputNodes;
  std::vector<Node *> outputNodes;

public:
  TruncSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::TruncIOp, handshake::TruncFOp>(
            [&](auto truncOp) {
              inputWidth = handshake::getHandshakeTypeBitWidth(truncOp.getOperand().getType());
              outputWidth = handshake::getHandshakeTypeBitWidth(truncOp.getResult().getType());
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
      if (nodeName.find("ins") != std::string::npos) {
        inputNodes.push_back(node);
      } else if (nodeName.find("outs") != std::string::npos) {
        outputNodes.push_back(node);
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }
};

class OperationDifferentiator {
  static DenseMap<Operation *, OperationDifferentiator> moduleMap;
  Operation *op;
  /// MLIR context to create attributes with.
  SmallVector<Operation *> inputs;
  SmallVector<Operation *> outputs;
  std::string fullPath = "/home/oyasar/full_integration/blif_files/";
  std::string moduleType;
  std::string uniqueName;
  // Follows the same naming convention as in the HDL files
  unsigned int size = 0;
  unsigned int selectType = 0;
  unsigned int indexType = 0;
  unsigned int addrType = 0;
  unsigned int dataWidth = 0;
  unsigned int inputWidth = 0;
  unsigned int outputWidth = 0;
  BlifData *blifData;

  void appendVarsToPath(std::initializer_list<unsigned int> inputs) {
    fullPath += moduleType + "/";
    for (int input : inputs) {
      fullPath += std::to_string(input) + "/";
    }
    fullPath += moduleType + ".blif";
  }

public:
  OperationDifferentiator(Operation *op) : op(op) {
    llvm::outs() << "Module name: " << op->getName() << "\n";
    llvm::outs() << "Module Unique Name: " << getUniqueName(op) << "\n";
    moduleType = op->getName().getStringRef();
    uniqueName = getUniqueName(op);
    // Find the position of the dot
    size_t dotPosition = moduleType.find('.');
    if (dotPosition != std::string::npos) {
      // Extract the part after the dot in handshake.modulename
      moduleType = moduleType.substr(dotPosition + 1);
    } else {
      llvm::errs() << "No dot found in the string."
                   << "\n";
    }
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::InstanceOp>([&](handshake::InstanceOp instOp) {
          // op->emitRemark("Instance Op");
        })
        .Case<handshake::ForkOp, handshake::LazyForkOp>([&](auto) {
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
        .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
          // Number of input data channels, data bitwidth, and select bitwidth
          size = muxOp.getDataOperands().size();
          dataWidth =
              handshake::getHandshakeTypeBitWidth(muxOp.getResult().getType());
          selectType = handshake::getHandshakeTypeBitWidth(
              muxOp.getSelectOperand().getType());
          appendVarsToPath({size, dataWidth, selectType});
        })
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
                appendVarsToPath({size, selectType});
              } else {
                appendVarsToPath({size, dataWidth, selectType});
              }
            })
        .Case<handshake::MergeOp>([&](auto) {
          // Number of input data channels and data bitwidth
          size = op->getNumOperands();
          dataWidth =
              handshake::getHandshakeTypeBitWidth(op->getResult(0).getType());
          if (dataWidth == 0) {
            moduleType += "_dataless";
            appendVarsToPath({size});
          } else {
            appendVarsToPath({size, dataWidth});
          }
        })
        .Case<handshake::JoinOp>([&](auto) {
          // Number of input channels
          size = op->getNumOperands();
          appendVarsToPath({size});
        })
        .Case<handshake::BranchOp, handshake::SinkOp, handshake::BufferOp>(
            [&](auto) {
              dataWidth = handshake::getHandshakeTypeBitWidth(
                  op->getOperand(0).getType());
              if (dataWidth == 0) {
                moduleType += "_dataless";
                appendVarsToPath({});
              } else {
                appendVarsToPath({dataWidth});
              }
            })
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
        .Case<handshake::SourceOp>([&](auto) {
          // No discrimianting parameters, just to avoid falling into the
          // default case for sources
          appendVarsToPath({});
        })
        .Case<handshake::LoadOpInterface>(
            [&](handshake::LoadOpInterface loadOp) {
              // Data bitwidth and address bitwidth
              dataWidth = handshake::getHandshakeTypeBitWidth(
                  loadOp.getDataInput().getType());
              addrType = handshake::getHandshakeTypeBitWidth(
                  loadOp.getAddressInput().getType());
              appendVarsToPath({addrType, dataWidth});
            })
        .Case<handshake::StoreOpInterface>(
            [&](handshake::StoreOpInterface storeOp) {
              // Data bitwidth and address bitwidth
              dataWidth = handshake::getHandshakeTypeBitWidth(
                  storeOp.getDataInput().getType());
              addrType = handshake::getHandshakeTypeBitWidth(
                  storeOp.getAddressInput().getType());
              appendVarsToPath({addrType, dataWidth});
            })
        .Case<handshake::SharingWrapperOp>(
            [&](handshake::SharingWrapperOp sharingWrapperOp) {
              assert(false && "operation unsupported");
            })
        .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
          // Bitwidth and binary-encoded constant value
          handshake::ChannelType cstType = cstOp.getResult().getType();
          unsigned bitwidth = cstType.getDataBitWidth();
          dataWidth = bitwidth;
          appendVarsToPath({dataWidth});

          if (bitwidth > 64) {
            cstOp.emitError() << "Constant value has bitwidth " << bitwidth
                              << ", but we only support up to 64.";
            return;
          }
          // TODO: add constant value
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
              handshake::SubIOp, handshake::XOrIOp>([&](auto) {
          // Bitwidth
          dataWidth =
              handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
          appendVarsToPath({dataWidth});
        })
        .Case<handshake::SelectOp>([&](handshake::SelectOp selectOp) {
          // Data bitwidth
          dataWidth = handshake::getHandshakeTypeBitWidth(
              selectOp.getTrueValue().getType());
          llvm::errs() << "SelectOp data width: "
                       << "\n";
        })
        .Case<handshake::CmpIOp>([&](handshake::CmpIOp cmpIOp) {
          // Predicate and bitwidth
          dataWidth =
              handshake::getHandshakeTypeBitWidth(cmpIOp.getLhs().getType());

          appendVarsToPath({dataWidth});
          // addString("PREDICATE", stringifyEnum(cmpIOp.getPredicate()));
        })
        .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::TruncIOp,
              handshake::ExtFOp, handshake::TruncFOp>([&](auto) {
          // Input bitwidth and output bitwidth
          inputWidth =
              handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
          outputWidth =
              handshake::getHandshakeTypeBitWidth(op->getResult(0).getType());
          appendVarsToPath({inputWidth, outputWidth});
        })
        .Default(
            [&](auto) {
              op->emitError()
                  << "No subject graph can be generated for this operation";
              return;
            });

    llvm::outs() << "Module Attributes:\n";
    llvm::outs() << "Module Type: " << moduleType << "\n";
    llvm::outs() << "Unique Name: " << uniqueName << "\n";
    llvm::outs() << "Size: " << size << "\n";
    llvm::outs() << "Select Type: " << selectType << "\n";
    llvm::outs() << "Index Type: " << indexType << "\n";
    llvm::outs() << "Address Type: " << addrType << "\n";
    llvm::outs() << "Data Width: " << dataWidth << "\n";
    llvm::outs() << "Input Width: " << inputWidth << "\n";
    llvm::outs() << "Output Width: " << outputWidth << "\n";
    llvm::outs() << "Full Path: " << fullPath << "\n";

    experimental::BlifParser parser;
    blifData = parser.parseBlifFile(fullPath);

    for (auto &node : blifData->getAllNodes()) {
      auto nodeName = node->getName();
      if (nodeName.find("result") != std::string::npos ||
          nodeName.find("outs") != std::string::npos) {
        node->setName(uniqueName + "_" + nodeName);
      } else if (nodeName.find(".") != std::string::npos) {
        node->setName(uniqueName + "." + nodeName);
      }
    }

    blifData->generateBlifFile(
        "/home/oyasar/full_integration/dynamatic_generated/" + uniqueName +
        ".blif");
  }
};

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_SUBJECT_GRAPH_H