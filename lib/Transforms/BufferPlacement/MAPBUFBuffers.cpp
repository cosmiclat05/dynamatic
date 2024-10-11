//===- MAPBUFBuffers.cpp - MAPBUF buffer placement -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements MAPBUF smart buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/MAPBUFBuffers.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/BlifReader.h"
#include "experimental/Support/CutEnumeration.h"
#include "gurobi_c.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <boost/functional/hash/extensions.hpp>
#include <list>
#include <omp.h>
#include <string>
#include <unordered_map>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::mapbuf;

MAPBUFBuffers::MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, StringRef blifFile)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod),
      blifFile(blifFile) {
  if (!unsatisfiable)
    setup();
}

MAPBUFBuffers::MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, StringRef blifFile,
                             Logger &logger, StringRef milpName)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                          milpName),
      blifFile(blifFile) {
  if (!unsatisfiable)
    setup();
}

void MAPBUFBuffers::extractResult(BufferPlacement &placement) {
  // Iterate over all channels in the circuit
  for (auto [channel, channelVars] : vars.channelVars) {
    // Extract number and type of slots from the MILP solution, as well as
    // channel-specific buffering properties

    unsigned numSlotsToPlace = static_cast<unsigned>(
        channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    if (numSlotsToPlace == 0)
      continue;

    bool placeOpaque = channelVars.signalVars[SignalType::DATA].bufPresent.get(
                           GRB_DoubleAttr_X) > 0;
    bool placeTransparent =
        channelVars.signalVars[SignalType::READY].bufPresent.get(
            GRB_DoubleAttr_X) > 0;

    handshake::ChannelBufProps &props = channelProps[channel];
    PlacementResult result;
    if (placeOpaque && placeTransparent) {
      // Place at least one opaque slot and satisfy the opaque slot requirement,
      // all other slots are transparent
      result.numOpaque = std::max(props.minOpaque, 1U);
      result.numTrans = numSlotsToPlace - result.numOpaque;
    } else if (placeOpaque) {
      // Satisfy the transparent slots requirement, all other slots are opaque
      result.numTrans = props.minTrans;
      result.numOpaque = numSlotsToPlace - props.minTrans;
    } else {
      // All slots transparent
      result.numTrans = numSlotsToPlace;
    }

    result.deductInternalBuffers(Channel(channel), timingDB);
    placement[channel] = result;
  }

  if (logger)
    logResults(placement);
}

void MAPBUFBuffers::addCustomChannelConstraints(Value channel) {
  // Get channel-specific buffering properties and channel's variables
  handshake::ChannelBufProps &props = channelProps[channel];
  ChannelVars &chVars = vars.channelVars[channel];

  // Force buffer presence if at least one slot is requested
  unsigned minSlots = props.minOpaque + props.minTrans;
  if (minSlots > 0) {
    model.addConstr(chVars.bufPresent == 1, "custom_forceBuffers");
    model.addConstr(chVars.bufNumSlots >= minSlots, "custom_minSlots");
  }

  // Set constraints based on minimum number of buffer slots
  GRBVar &bufData = chVars.signalVars[SignalType::DATA].bufPresent;
  GRBVar &bufReady = chVars.signalVars[SignalType::READY].bufPresent;
  if (props.minOpaque > 0) {
    // Force the MILP to place at least one opaque slot
    model.addConstr(bufData == 1, "custom_forceData");
    // If the MILP decides to also place a ready buffer, then we must reserve
    // an extra slot for it
    model.addConstr(chVars.bufNumSlots >= props.minOpaque + bufReady,
                    "custom_minData");
  }
  if (props.minTrans > 0) {
    // Force the MILP to place at least one transparent slot
    model.addConstr(bufReady == 1, "custom_forceReady");
    // If the MILP decides to also place a data buffer, then we must reserve
    // an extra slot for it
    model.addConstr(chVars.bufNumSlots >= props.minTrans + bufData,
                    "custom_minReady");
  }

  // Set constraints based on maximum number of buffer slots
  if (props.maxOpaque && props.maxTrans) {
    unsigned maxSlots = *props.maxOpaque + *props.maxTrans;
    // Forbid buffer placement on the channel entirely when no slots are allowed
    if (maxSlots == 0)
      model.addConstr(chVars.bufPresent == 0, "custom_noBuffer");
    // Restrict the maximum number of slots allowed
    model.addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
  }

  // Forbid placement of some buffer type based on maximum number of allowed
  // slots on each signal
  if (props.maxOpaque && *props.maxOpaque == 0) {
    // Force the MILP to use transparent slots only
    model.addConstr(bufData == 0, "custom_noData");
  }
  if (props.maxTrans && *props.maxTrans == 0) {
    // Force the MILP to use opaque slots only
    model.addConstr(bufReady == 0, "custom_noReady");
  }
}

void MAPBUFBuffers::addCutSelectionConstraints() {
  for (auto &[key, val] : experimental::Cuts::cuts) {
    GRBLinExpr cutSelectionSum = 0;

    // Use an indexed for loop to include i in the loop declaration
    for (size_t i = 0; i < val.size(); ++i) {
      auto &cut = val[i];
      GRBVar &cutSelection = cut.cutSelection;
      cutSelection =
          model.addVar(0, GRB_INFINITY, 0, GRB_BINARY,
                       (cut.root + "__CutSelection_" + std::to_string(i)));
      cutSelectionSum += cutSelection;
    }

    model.update();
    model.addConstr(cutSelectionSum == 1, "cut_selection_constraint");
  }
}

std::optional<GRBVar> variableExists(GRBModel &model,
                                     const std::string &varName) {
  GRBVar *vars = model.getVars();
  int numVars = model.get(GRB_IntAttr_NumVars);

  // Loop through all variables and check their names
  for (int i = 0; i < numVars; i++) {
    if (vars[i].get(GRB_StringAttr_VarName) == varName) {
      return vars[i]; // Variable exists
    }
  }
  return {}; // Variable does not exist
}

static StringRef getSignalName(SignalType type) {
  switch (type) {
  case SignalType::DATA:
    return "data";
  case SignalType::VALID:
    return "valid";
  case SignalType::READY:
    return "ready";
  }
}

enum ConstraintNames {
  trivialCut_noBuf,       // 0
  trivialCut_Buf,         // 1
  oneCut_noBuf,           // 2
  oneCut_Buf,             // 3
  moreCut_noBuf,          // 4
  moreCut_Buf,            // 5
  cut_selection_conflict, // 6
  equality                // 7

};

constexpr std::string_view getConstraintName(ConstraintNames constraint) {
  switch (constraint) {
  case oneCut_noBuf:
    return "oneCut_noBusssf";
  case oneCut_Buf:
    return "oneCut_Buf";
  case moreCut_noBuf:
    return "moreCut_noBuf";
  case moreCut_Buf:
    return "moreCut_Buf";
  case trivialCut_noBuf:
    return "trivialCut_noBuf";
  case trivialCut_Buf:
    return "trivialCut_Buf";
  case cut_selection_conflict:
    return "cut_selection_conflict";
  case equality:
    return "equality";
  }
}

struct GrbConst {
  GRBLinExpr lhs;
  GRBLinExpr rhs;
  ConstraintNames constraintType;
  GrbConst(const GRBLinExpr &lhs, const GRBLinExpr &rhs, int type)
      : lhs(lhs), rhs(rhs), constraintType(static_cast<ConstraintNames>(type)) {
  }
};

class VariableSearcher {
private:
  std::unordered_map<std::string, std::vector<GRBVar>> variableMap;

public:
  VariableSearcher(const std::vector<GRBVar> &channelVarsVec) {
    for (const auto &var : channelVarsVec) {
      std::string varName = var.get(GRB_StringAttr_VarName);
      variableMap[varName].push_back(var);
    }
  }

  GRBVar variableIncludes(const std::string &subStr,
                          std::unordered_map<std::string, GRBVar> &nodeToGrb,
                          const std::string &key) {
    for (auto &pair : variableMap) {
      if (pair.first.find(subStr) != std::string::npos) {
        return pair.second.front();
      }
    }
    return nodeToGrb[key];
  }

  std::optional<GRBVar> variableIncludes(const std::string &subStr) const {
    for (const auto &pair : variableMap) {
      if (pair.first.find(subStr) != std::string::npos) {
        return pair.second.front();
      }
    }
    return {};
  }

  std::vector<GRBVar> variableVectors(const std::string &subStr) const {
    std::vector<GRBVar> result;
    for (const auto &pair : variableMap) {
      if (pair.first.find(subStr) != std::string::npos) {
        result.insert(result.end(), pair.second.begin(), pair.second.end());
      }
    }
    return result;
  }
};

bool isChannelVar(const std::string &node) {
  // a hacky way to determine if a variable is a channel variable.
  // if it includes "new", "." and does not include "_", it is not a channel
  // variable
  return node.find("new") == std::string::npos &&
         node.find('.') == std::string::npos &&
         node.find('_') != std::string::npos;
}

std::ostringstream retrieveChannelName(const std::string &node,
                                       const std::string &variableType) {
  if (!isChannelVar(node)) {
    return std::ostringstream{node};
  }

  std::string variableTypeName;
  if (variableType == "buffer")
    variableTypeName = "BufPresent_";
  else if (variableType == "pathIn")
    variableTypeName = "PathIn_";
  else if (variableType == "pathOut")
    variableTypeName = "PathOut_";

  std::stringstream ss(node);
  std::string token;
  std::vector<std::string> result;

  while (std::getline(ss, token, '_')) {
    result.emplace_back(token);
  }

  const auto leafLastUnderscore = node.find_last_of('_');
  const auto leafNodeNameTillUnderScore = node.substr(0, leafLastUnderscore);
  const auto channelTypeName = node.substr(leafLastUnderscore + 1);

  std::ostringstream bufferVarNameStream;
  if (result.back() == "valid" || result.back() == "ready") {
    bufferVarNameStream << (result.back() == "valid"
                                ? ("valid" + variableTypeName)
                                : ("ready" + variableTypeName))
                        << leafNodeNameTillUnderScore;
  } else {
    const auto leafLastPar = node.find_last_of('[');
    const auto leafNodeNameTillPar = node.substr(0, leafLastPar);
    bufferVarNameStream << ("data" + variableTypeName) << leafNodeNameTillPar;
  }

  return bufferVarNameStream;
}

std::ostringstream
retrieveBlackboxInputChannel(const std::string &node,
                             const std::string &variableType) {
  if (!isChannelVar(node)) {
    return std::ostringstream{node};
  }

  std::string variableTypeName;
  if (variableType == "buffer")
    variableTypeName = "BufPresent_";
  else if (variableType == "pathIn")
    variableTypeName = "PathIn_";
  else if (variableType == "pathOut")
    variableTypeName = "PathOut_";

  std::stringstream ss(node);
  std::string token;
  std::vector<std::string> result;

  while (std::getline(ss, token, '_')) {
    result.emplace_back(token);
  }

  const auto leafLastUnderscore = node.find_last_of('_');
  const auto leafNodeNameTillUnderScore = node.substr(0, leafLastUnderscore);
  const auto channelTypeName = node.substr(leafLastUnderscore + 1);

  std::ostringstream bufferVarNameStream;
  if (result.back() == "valid" || result.back() == "ready") {
    bufferVarNameStream << (result.back() == "valid"
                                ? ("valid" + variableTypeName)
                                : ("ready" + variableTypeName))
                        << leafNodeNameTillUnderScore;
  } else {
    const auto leafLastPar = node.find_last_of('[');
    const auto leafNodeNameTillPar = node.substr(0, leafLastPar);
    bufferVarNameStream << ("data" + variableTypeName) << leafNodeNameTillPar;
  }

  return bufferVarNameStream;
}

void MAPBUFBuffers::addCutLoopbackBuffers() {
  auto funcOp = funcInfo.funcOp;
  funcOp.walk([&](mlir::Operation *op) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (isBackedge(result, user)) {
          handshake::ChannelBufProps &resProps = channelProps[op->getResult(0)];
          if (resProps.maxTrans.value_or(1) >= 1) {
            GRBVar &bufVar = vars.channelVars[op->getResult(0)]
                                 .signalVars[SignalType::READY]
                                 .bufPresent;
            model.addConstr(bufVar == 0, "backedge_ready");
          } 
          if (resProps.maxOpaque.value_or(1) >= 1) {
            GRBVar &bufVar = vars.channelVars[op->getResult(0)]
                                 .signalVars[SignalType::DATA]
                                 .bufPresent;
            model.addConstr(bufVar == 0, "backedge_data");
          } 
        }
      }
    }
  });
}

void MAPBUFBuffers::addBlackboxConstraints() {
  for (auto &[channel, _] : channelProps) {
    Operation *definingOp = channel.getDefiningOp();
    if (!definingOp) {
      continue;
    }

    auto numOps = definingOp->getNumOperands();
    for (unsigned int i = 0; i < numOps; i++) {
      auto isSubi = getUniqueName(definingOp).find("subi") != std::string::npos;
      auto isAddi = getUniqueName(definingOp).find("addi") != std::string::npos;
      auto isCmpi = getUniqueName(definingOp).find("cmpi") != std::string::npos;

      if (isSubi || isAddi || isCmpi) {

        unsigned int bitwidth =
            handshake::getHandshakeTypeBitWidth(channel.getType());
        if (isCmpi && bitwidth <= 4) {
          continue;
        }

        ChannelVars &channelVars = vars.channelVars[channel];
        Value definingChannel = definingOp->getOperand(i);
        ChannelSignalVars &validVars =
            channelVars.signalVars[SignalType::VALID];
        GRBVar &adderOutValidPathIn = validVars.path.tIn;

        GRBVar &adderInValid = vars.channelVars[definingChannel]
                                   .signalVars[SignalType::VALID]
                                   .path.tOut;

        ChannelSignalVars &readyVars =
            channelVars.signalVars[SignalType::READY];
        GRBVar &adderOutReadyPathOut = readyVars.path.tOut;

        GRBVar &adderInReady = vars.channelVars[definingChannel]
                                   .signalVars[SignalType::READY]
                                   .path.tIn;

        model.addConstr(adderOutReadyPathOut == adderInReady,
                        "adderConstraint_ready");

        model.addConstr(adderInValid == adderOutValidPathIn,
                        "adderConstraint_valid");

        ChannelSignalVars &dataVars = channelVars.signalVars[SignalType::DATA];
        GRBVar &adderOutPathIn = dataVars.path.tIn;

        GRBVar &adderInData = vars.channelVars[definingChannel]
                                  .signalVars[SignalType::DATA]
                                  .path.tOut;

        std::map<unsigned int, double> adderDelay = {{1, 0.587},  {2, 0.587},
                                                     {4, 0.993},  {8, 0.991},
                                                     {16, 1.099}, {32, 1.315}};

        std::map<unsigned int, double> compDelay = {{1, 0.587}, {2, 0.587},
                                                    {4, 0.993}, {8, 0.6},
                                                    {16, 0.8},  {32, 1.005}};

        double delay = 0.0;

        if (getUniqueName(definingOp).find("cmpi") != std::string::npos) {
          // Find the smallest key in compDelay that is greater than or equal to
          // bitwidth
          auto it = compDelay.lower_bound(bitwidth);
          if (it == compDelay.end() || it->first != bitwidth) {
            // If bitwidth is not exactly a key, use the next higher key
            it = compDelay.upper_bound(bitwidth);
          }
          if (it != compDelay.end()) {
            delay = it->second;
          }
          model.addConstr(adderInData + delay == adderOutPathIn,
                          "adderConstraint_" + std::to_string(bitwidth));

          continue;
        }

        // Find the smallest key in adderDelay that is greater than or equal
        // to bitwidth
        auto it = adderDelay.lower_bound(bitwidth);
        if (it == adderDelay.end() || it->first != bitwidth) {
          // If bitwidth is not exactly a key, use the next higher key
          it = adderDelay.upper_bound(bitwidth);
        }
        if (it != adderDelay.end()) {
          delay = it->second;
        }

        model.addConstr(adderInData + delay == adderOutPathIn,
                        "adderConstraint_" + std::to_string(bitwidth));
      }
    }
  }
  model.update();
}

// void MAPBUFBuffers::addClockPeriodConstraints(
//     experimental::BlifData &blif,
//     std::unordered_map<std::string, GRBVar> &nodeToGRB, VariableSearcher& pathInSearcher) {
//   for (auto &node : blif.getNodes()) {
//     GRBVar &nodeVar = nodeToGRB[node];
//     nodeVar = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, node);
//     model.update();
//     if (blif.isPrimaryInput(node)) {
//       model.addConstr(nodeVar == 0, "input_delay");
//       std::string pathInName = retrieveChannelName(node, "pathIn").str();
//       if (blif.isConstantNode(node)) {
//         std::string pathInName = retrieveChannelName(node, "pathIn").str();
//         GRBVar nodeVarChannel =
//             pathInSearcher.variableIncludes(pathInName, nodeToGRB, node);

//         model.addConstr(nodeVar == nodeVarChannel, "primary_input_channel");
//       }
//     } else {
//       model.addConstr(nodeVar <= targetPeriod, "clock_period_constraint");
//     }
//   }
//   model.update();
// }

void MAPBUFBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signals;
  signals.push_back(SignalType::DATA);
  signals.push_back(SignalType::VALID);
  signals.push_back(SignalType::READY);

  experimental::BlifParser parser;
  experimental::BlifData anchorsRemoved =
      parser.parseBlifFile(blifFile.str() + "no_anchors.blif");
  experimental::BlifData withAnchors =
      parser.parseBlifFile(blifFile.str() + "anchored_strashed.blif");

  experimental::Cuts cutsAnchorsRemoved(anchorsRemoved, 6, 0);
  cutsAnchorsRemoved.runCutAlgos(false, true, false, false);

  experimental::Cuts cutsWithAnchors(withAnchors, 6, 0);
  cutsWithAnchors.runCutAlgos(true, true, false, true);
  experimental::Cuts::printCuts("cuts.txt");

  /// NOTE: (lucas-rami) For each buffering group this should be the timing
  /// model of the buffer that will be inserted by the MILP for this group. We
  /// don't have models for these buffers at the moment therefore we provide a
  /// null-model to each group, but this hurts our placement's accuracy.
  const TimingModel *bufModel = nullptr;

  BufferingGroup dataValidGroup({SignalType::DATA, SignalType::VALID},
                                bufModel);
  BufferingGroup readyGroup({SignalType::READY}, bufModel);

  SmallVector<BufferingGroup> bufGroups;
  bufGroups.push_back(dataValidGroup);
  bufGroups.push_back(readyGroup);

  std::unordered_map<std::string, GRBVar> nodeToGRB;
  std::unordered_map<std::string, GRBVar> channelToGRB;

  std::vector<Value> allChannels;
  std::vector<GRBVar> bufVarsVector;
  std::vector<GRBVar> pathInVarsVector;
  std::vector<GRBVar> pathOutVarsVector;

  GRBVar clockVar = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "clock");
  model.addConstr(clockVar == targetPeriod, "clock_period");

  for (auto &[channel, _] : channelProps) {
    // Create channel variables and constraints
    allChannels.push_back(channel);
    addChannelVars(channel, signals);
    addCustomChannelConstraints(channel);
    ChannelVars &channelVars = vars.channelVars[channel];

    GRBVar &bufPresent = channelVars.bufPresent;
    GRBVar &bufNumSlots = channelVars.bufNumSlots;
    GRBVar &bufVarData =
        vars.channelVars[channel].signalVars[SignalType::DATA].bufPresent;
    GRBVar &bufVarReady =
        vars.channelVars[channel].signalVars[SignalType::READY].bufPresent;

    model.addConstr(bufVarData + bufVarReady == bufNumSlots, "buf_slots");

    model.addConstr(bufVarData <= bufVarReady, "buf_order");

    // if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
    //     !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
    for (auto &signal : signals) {
      std::string suffix = "_" + getUniqueName(*channel.getUses().begin());
      ChannelSignalVars &signalVars = channelVars.signalVars[signal];
      GRBVar &bufVarSignal =
          vars.channelVars[channel].signalVars[signal].bufPresent;
      GRBVar &t1 = signalVars.path.tIn;
      GRBVar &t2 = signalVars.path.tOut;
      model.addConstr(t2 <= clockVar, "pathOut_period");
      model.addConstr(t1 <= clockVar, "pathIn_period");
      model.addConstr(t2 - t1 + 100 * bufVarSignal >= 0, "buf_delay");

      pathInVarsVector.push_back(t1);
      pathOutVarsVector.push_back(t2);
      bufVarsVector.push_back(bufVarSignal);
    }
    addChannelElasticityConstraints(channel, bufGroups);
    // }
  }

  // ChannelFilter channelFilter = [&](Value channel) -> bool {
  //   Operation *defOp = channel.getDefiningOp();
  //   return !isa_and_present<handshake::MemoryOpInterface>(defOp) &&
  //          !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin());
  // };

  // for (Operation &op : funcInfo.funcOp.getOps()) {
  //   // addUnitElasticityConstraints(&op, channelFilter);
  // }

 // addClockPeriodConstraints(anchorsRemoved, nodeToGRB);

  addBlackboxConstraints();

  addCutLoopbackBuffers();

  addCutSelectionConstraints();

  VariableSearcher bufSearcher(bufVarsVector);
  VariableSearcher pathInSearcher(pathInVarsVector);
  VariableSearcher pathOutSearcher(pathOutVarsVector);

  for (auto &node : anchorsRemoved.getNodes()) {
    GRBVar &nodeVar = nodeToGRB[node];
    nodeVar = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, node);
    model.update();
    if (anchorsRemoved.isPrimaryInput(node)) {
      model.addConstr(nodeVar == 0, "primary_input");

      // if (anchorsRemoved.isConstantNode(node)) {
      //   std::string pathInName = retrieveChannelName(node, "pathIn").str();
      //   GRBVar nodeVarChannel =
      //       pathInSearcher.variableIncludes(pathInName, nodeToGRB, node);

      //   model.addConstr(nodeVar == nodeVarChannel, "primary_input_channel");
      // }
    } else {
      model.addConstr(nodeVar <= clockVar, "clock_period_constraint");
    }
  }
  model.update();

  std::list<GrbConst> constraints;
  std::list<GrbConst> constraintsEqual;

  int n = experimental::Cuts::cuts.size();

  std::unordered_map<std::pair<std::string, std::string>,
                     std::vector<std::string>,
                     boost::hash<std::pair<std::string, std::string>>>
      leafToRootPaths;

  std::ifstream file(blifFile.str() + "./fpga20_results.txt");
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string varName;
    double value;
    if (iss >> varName >> value) {
      // Retrieve the Gurobi variable corresponding to varName
      auto varOpt = variableExists(model, varName);
      if (!varOpt) {
        llvm::errs() << "Variable " << varName << " not found in the model."
                     << "\n";
        continue;
      }
      GRBVar var = varOpt.value();
      // Add the equality constraint
      model.addConstr(var == value, "fpga20_results");
    }
  }
  file.close();

  for (int i = 0; i < n; ++i) {
    auto it = std::next(experimental::Cuts::cuts.begin(), i);
    const auto &key = it->first;
    auto &val = it->second;

    std::string pathInName = retrieveChannelName(key, "pathIn").str();
    GRBVar nodeVar =
        pathInSearcher.variableIncludes(pathInName, nodeToGRB, key);

    for (auto &cut : val) {
      GRBVar &cutSelectionVar = cut.cutSelection;
      if ((cut.leaves.size() == 1) && (*cut.leaves.begin() == key)) {
        std::set<std::string> fanIns = anchorsRemoved.getFanins(key);
        for (auto &fanIn : fanIns) {
          std::string pathOutName = retrieveChannelName(fanIn, "pathOut").str();
          GRBVar faninVar =
              pathOutSearcher.variableIncludes(pathOutName, nodeToGRB, fanIn);
          if (fanIns.size() == 1) {
            model.addConstr(nodeVar == faninVar, "trivial_single_fanin");
            break;
          }
          model.addConstr(nodeVar + (1 - cutSelectionVar) * 100 >=
                              faninVar + 0.6,
                          "trivial_multiple_fanin");
        }
        continue;
      }

      for (auto &leaf : cut.leaves) {
        std::string pathOutName = retrieveChannelName(leaf, "pathOut").str();
        GRBVar leafVar =
            pathOutSearcher.variableIncludes(pathOutName, nodeToGRB, leaf);

        model.addConstr(nodeVar + (1 - cutSelectionVar) * 100 >= leafVar + 0.6,
                        "delay_propagation");

        std::vector<std::string> path;
        auto leafKeyPair = std::make_pair(leaf, key);
        {
          if (leafToRootPaths.find(leafKeyPair) != leafToRootPaths.end()) {
            path = leafToRootPaths[leafKeyPair];
          } else {
            path = anchorsRemoved.findPath(leaf, key);
            if (!path.empty()) {
              // remove the node itself from the path, it messes up with cut
              // selection conflicts
              path.pop_back();
              // remove also the starting node, because we should be able to
              // place a buffer there
              path.erase(path.begin());
            }
            leafToRootPaths[leafKeyPair] = path;
          }
        }
        std::vector<std::string> nodesWithChannels;
        for (auto &nodePath : path) {
          std::string nodeWithChannel =
              retrieveChannelName(nodePath, "buffer").str();
          if (nodeWithChannel != nodePath) {
            auto bufferVars = bufSearcher.variableIncludes(nodeWithChannel);
            GRBVar bufferLeaf;
            if (bufferVars) {
              bufferLeaf = bufferVars.value();

              model.addConstr(1 >= bufferLeaf + cutSelectionVar,
                              "cut_selection_conflict");
            }
          }
        }
      }
    }
  }

  model.update();

  SmallVector<CFDFC *> cfdfcs;
  for (auto [cfdfc, optimize] : funcInfo.cfdfcs) {
    if (!optimize)
      continue;
    cfdfcs.push_back(cfdfc);
    addCFDFCVars(*cfdfc);
    addChannelThroughputConstraints(*cfdfc);
    addUnitThroughputConstraints(*cfdfc);
  }

  // unsigned totalExecs = 0;
  // for (Value channel : allChannels) {
  //   totalExecs += getChannelNumExecs(channel);
  // }

  // // Create the expression for the MILP objective
  // GRBLinExpr objective;

  // // For each CFDFC, add a throughput contribution to the objective, weighted
  // // by the "importance" of the CFDFC
  // double maxCoefCFDFC = 0.0;
  // double fTotalExecs = static_cast<double>(totalExecs);
  // if (totalExecs != 0) {
  //   for (CFDFC *cfdfc : cfdfcs) {
  //     double coef = (cfdfc->channels.size() * cfdfc->numExecs) / fTotalExecs;
  //     objective += coef * vars.cfVars[cfdfc].throughput;
  //     maxCoefCFDFC = std::max(coef, maxCoefCFDFC);
  //   }
  // }

  // // In case we ran the MILP without providing any CFDFC, set the maximum CFDFC
  // // coefficient to any positive value
  // if (maxCoefCFDFC == 0.0)
  //   maxCoefCFDFC = 1.0;

  // // For each channel, add a "penalty" in case a buffer is added to the channel,
  // // and another penalty that depends on the number of slots
  // double bufPenaltyMul = 1e-4;
  // double slotPenaltyMul = 1e-5;
  // for (Value channel : allChannels) {
  //   ChannelVars &channelVars = vars.channelVars[channel];
  //   objective -= maxCoefCFDFC * bufPenaltyMul * channelVars.bufPresent;
  //   objective -= maxCoefCFDFC * slotPenaltyMul * channelVars.bufNumSlots;
  // }

  // objective -= bufPenaltyMul * clockVar;
  // //  Finally, set the MILP objective
  // model.setObjective(objective, GRB_MAXIMIZE);
  addObjective(allChannels, cfdfcs);
  model.set(GRB_IntParam_DualReductions, 0);

  llvm::errs() << "model marked ready to optimize"
               << "\n";
  markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
