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
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
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
      // Place the minumum number of opaque slots; at least one and enough to
      // satisfy all our opaque/transparent requirements
      if (props.maxTrans) {
        // We must place enough opaque slots as to not exceed the maximum number
        // of transparent slots
        result.numOpaque =
            std::max(props.minOpaque, numSlotsToPlace - *props.maxTrans);
      } else {
        // At least one slot, but no more than necessary
        result.numOpaque = std::max(props.minOpaque, 1U);
      }
      // All remaining slots are transparent
      result.numTrans = numSlotsToPlace - result.numOpaque;
    } else if (placeOpaque) {
      // Place the minimum number of transparent slots; at least the expected
      // minimum and enough to satisfy all our opaque/transparent requirements
      if (props.maxOpaque) {
        result.numTrans =
            std::max(props.minTrans, numSlotsToPlace - *props.maxOpaque);
      } else {
        result.numTrans = props.minTrans;
      }
      // All remaining slots are opaque
      result.numOpaque = numSlotsToPlace - result.numTrans;
    } else {
      // placeOpaque == 0 --> props.minOpaque == 0 so all slots can be
      // transparent
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
  // Add constraints to ensure that only one cut is selected for each node.
  for (auto &[key, val] : experimental::Cuts::cuts) {
    GRBLinExpr cutSelectionSum = 0;
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

enum ConstraintNames {
  single_fanin_delay,     // 0
  trivial_cut_delay,      // 1
  delay_propagation,      // 2
  cut_selection_conflict, // 3

};

constexpr std::string_view getConstraintName(ConstraintNames constraint) {
  switch (constraint) {
  case single_fanin_delay:
    return "single_fanin_delay";
  case trivial_cut_delay:
    return "trivial_cut_delay";
  case delay_propagation:
    return "delay_propagation";
  case cut_selection_conflict:
    return "cut_selection_conflict";
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

void MAPBUFBuffers::addCutLoopbackBuffers() {
  // Add constraints to ensure that the loopback buffers are placed. Simply loop
  // over all the channels and check if the channel is a back edge.
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
            model.addConstr(bufVar == 1, "backedge_ready");
          }
          if (resProps.maxOpaque.value_or(1) >= 1) {
            GRBVar &bufVar = vars.channelVars[op->getResult(0)]
                                 .signalVars[SignalType::DATA]
                                 .bufPresent;
            model.addConstr(bufVar == 1, "backedge_data");
          }
        }
      }
    }
  });
}

void MAPBUFBuffers::addBlackboxConstraints() {
  // Add constraints for blackbox operations, addi, subi, cmpi and
  // mem_controller. These delays are retrieved from Vivado Timing Reports.
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
      auto isMem =
          getUniqueName(definingOp).find("mem_controller") != std::string::npos;

      if (isSubi || isAddi || isCmpi || isMem) {
        Value inputChannel = definingOp->getOperand(i);
        ChannelVars &inputChannelVars = vars.channelVars[inputChannel];
        ChannelVars &outputChannelVars = vars.channelVars[channel];
        unsigned int bitwidth = 0;

        if (isMem)
          continue;

        if (!isMem) {
          bitwidth =
              handshake::getHandshakeTypeBitWidth(inputChannel.getType());

          if (bitwidth <= 4) {
            break;
          }
        }
        GRBVar &adderOutValid =
            outputChannelVars.signalVars[SignalType::VALID].path.tIn;

        GRBVar &adderInValid =
            inputChannelVars.signalVars[SignalType::VALID].path.tOut;

        GRBVar &adderOutReady =
            outputChannelVars.signalVars[SignalType::READY].path.tOut;

        GRBVar &adderInReady =
            inputChannelVars.signalVars[SignalType::READY].path.tIn;

        model.addConstr(adderOutReady + 0.1 == adderInReady,
                        "adderConstraint_ready");

        model.addConstr(adderInValid + 0.1 == adderOutValid,
                        "adderConstraint_valid");

        GRBVar &adderOutPathIn =
            outputChannelVars.signalVars[SignalType::DATA].path.tIn;

        GRBVar &adderInData =
            inputChannelVars.signalVars[SignalType::DATA].path.tOut;

        // std::map<unsigned int, double> adderDelay = {{1, 0.587},  {2, 0.587},
        //                                              {4, 0.993},  {8, 0.991},
        //                                              {16, 1.099},
        //                                              {32, 1.315}};

        std::map<unsigned int, double> adderDelay = {
            {1, 0.587}, {2, 0.587}, {4, 0.993}, {8, 0.6}, {16, 0.7}, {32, 1.0}};

        std::map<unsigned int, double> compDelay = {
            {1, 0.587}, {2, 0.587}, {4, 0.993}, {8, 0.8}, {16, 0.1}, {32, 1.2}};

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
                          "cmpiConstraint_" + std::to_string(bitwidth));

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

        // if (isMem) {
        //   delay = 0.75;
        // }

        model.addConstr(adderInData + delay == adderOutPathIn,
                        "adderConstraint_" + std::to_string(bitwidth));
      }
    }
  }
  model.update();
}

void MAPBUFBuffers::addClockPeriodConstraintsNodes(
    experimental::BlifData &blif,
    std::unordered_map<std::string, GRBVar> &nodeToGRB) {
  // Add clock period constraints for subject graph edges. For subject graph
  // edges, there is no need for 2 variables like channels, one at the input and
  // one at the output. We only need one variable for each node.
  // Also adds constraints for primary inputs and constants.

  for (auto &node : blif.getNodes()) {
    GRBVar &nodeVar = nodeToGRB[node];
    nodeVar = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, node);
    model.update();
    if (blif.isPrimaryInput(node)) {
      model.addConstr(nodeVar == 0, "input_delay");
      // std::string pathInName = retrieveChannelName(node, "pathIn").str();
      // if (blif.isConstantNode(node)) {
      //   std::string pathInName = retrieveChannelName(node, "pathIn").str();
      //   GRBVar nodeVarChannel =
      //       pathInSearcher.variableIncludes(pathInName, nodeToGRB, node);

      //   model.addConstr(nodeVar == nodeVarChannel, "primary_input_channel");
      // }
    } else {
      model.addConstr(nodeVar <= targetPeriod, "clock_period_constraint");
    }
  }
  model.update();
}

void MAPBUFBuffers::addClockPeriodConstraintsChannels(Value channel,
                                                      SignalType signal) {
  // Add clock period constraints for each channel. The delay is not propagated
  // through the channel if a buffer is present.
  ChannelVars &channelVars = vars.channelVars[channel];
  std::string suffix = "_" + getUniqueName(*channel.getUses().begin());
  ChannelSignalVars &signalVars = channelVars.signalVars[signal];
  GRBVar &bufVarSignal =
      vars.channelVars[channel].signalVars[signal].bufPresent;
  GRBVar &t1 = signalVars.path.tIn;
  GRBVar &t2 = signalVars.path.tOut;
  model.addConstr(t2 <= targetPeriod, "pathOut_period");
  model.addConstr(t1 <= targetPeriod, "pathIn_period");
  model.addConstr(t2 - t1 + bigConstant * bufVarSignal >= 0, "buf_delay");

  pathInVarsVector.push_back(t1);
  pathOutVarsVector.push_back(t2);
  bufVarsVector.push_back(bufVarSignal);
}

void MAPBUFBuffers::retrieveFPGA20Constraints(GRBModel &model) {
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
      llvm::errs() << "Adding constraint: " << varName << " == " << value
                   << "\n";
      model.addConstr(var == value, "fpga20_results");
    }
  }
  file.close();
}

using pathMap =
    std::unordered_map<std::pair<std::string, std::string>,
                       std::vector<std::string>,
                       boost::hash<std::pair<std::string, std::string>>>;

std::vector<std::string>
getOrCreateLeafToRootPath(const std::string &key, const std::string &leaf,
                          pathMap leafToRootPaths,
                          experimental::BlifData &anchorsRemoved) {
  // Check if the path from leaf to root has already been computed, if not then
  // compute it, if so then return it. Returns the shortest path by running BFS.
  auto leafKeyPair = std::make_pair(leaf, key);

  if (leafToRootPaths.find(leafKeyPair) != leafToRootPaths.end()) {
    return leafToRootPaths[leafKeyPair];
  }

  auto path = anchorsRemoved.findPath(leaf, key);
  if (!path.empty()) {
    // remove the starting node and the root node, as we should be able to place
    // buffers on channels adjacent to these nodes
    path.pop_back();          // remove root
    path.erase(path.begin()); // remove starting node
  }

  leafToRootPaths[leafKeyPair] = path;
  return path;
}

void MAPBUFBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signals;
  signals.push_back(SignalType::DATA);
  signals.push_back(SignalType::VALID);
  signals.push_back(SignalType::READY);

  experimental::BlifParser parser;
  experimental::BlifData anchorsRemoved =
      parser.parseBlifFile(blifFile.str() + "noAnchors.blif");
  experimental::BlifData withAnchors =
      parser.parseBlifFile(blifFile.str() + "anchored.blif");

  // Run cut enumeration on the both versions of the subject graph, with anchors and without anchors.
  // Cut enumeration with anchors give us the cuts such that every node is enumerated until the channels.
  // Cut enumeration without anchors give us the deepest cuts.
  experimental::Cuts cutsAnchorsRemoved(anchorsRemoved, 6, 0);
  cutsAnchorsRemoved.runCutAlgos(false, true, false, false);

  experimental::Cuts cutsWithAnchors(withAnchors, 6, 0);
  cutsWithAnchors.runCutAlgos(false, true, false, true);
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

  GRBVar clockVar = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "clock");
  model.addConstr(clockVar == targetPeriod, "clock_period");

  for (auto &[channel, _] : channelProps) {
    // Create channel variables and constraints
    allChannels.push_back(channel);
    addChannelVars(channel, signals);
    addCustomChannelConstraints(channel);
    for (SignalType signal : signals) {
      // add clock period constraints for each channel
      addClockPeriodConstraintsChannels(channel, signal);
    }
    addChannelElasticityConstraints(channel, bufGroups);
  }

  ChannelFilter channelFilter = [&](Value channel) -> bool {
    Operation *defOp = channel.getDefiningOp();
    return !isa_and_present<handshake::MemoryOpInterface>(defOp) &&
           !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin());
  };

  for (Operation &op : funcInfo.funcOp.getOps()) {
    addUnitElasticityConstraints(&op, channelFilter);
  }

  retrieveFPGA20Constraints(model);

  addClockPeriodConstraintsNodes(anchorsRemoved, nodeToGRB);

  addBlackboxConstraints();

  // addCutLoopbackBuffers();

  addCutSelectionConstraints();

  VariableSearcher bufSearcher(bufVarsVector);
  VariableSearcher pathInSearcher(pathInVarsVector);
  VariableSearcher pathOutSearcher(pathOutVarsVector);

  // for (auto &node : anchorsRemoved.getNodes()) {
  //   GRBVar &nodeVar = nodeToGRB[node];
  //   nodeVar = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, node);
  //   model.update();
  //   if (anchorsRemoved.isPrimaryInput(node)) {
  //     model.addConstr(nodeVar == 0, "primary_input");

  //     // if (anchorsRemoved.isConstantNode(node)) {
  //     //   std::string pathInName = retrieveChannelName(node,
  //     "pathIn").str();
  //     //   GRBVar nodeVarChannel =
  //     //       pathInSearcher.variableIncludes(pathInName, nodeToGRB, node);

  //     //   model.addConstr(nodeVar == nodeVarChannel,
  //     "primary_input_channel");
  //     // }
  //   } else {
  //     model.addConstr(nodeVar <= clockVar, "clock_period_constraint");
  //   }
  // }
  // model.update();

  std::list<GrbConst> constraints;
  std::list<GrbConst> constraintsEqual;

  int n = experimental::Cuts::cuts.size();

  pathMap leafToRootPaths;

  auto retrieveAndSearchGRBVar = [&](const std::string &node,
                                     const std::string &variableType,
                                     VariableSearcher &searcher) -> GRBVar {
    std::string channelName = retrieveChannelName(node, variableType).str();
    return searcher.variableIncludes(channelName, nodeToGRB, node);
  };

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < n; ++i) {
    // Create local constraints for each thread
    std::list<GrbConst> localConstraints;
    std::list<GrbConst> localConstraintsEqual;
    // Loop over each subject graph edge
    auto it = std::next(experimental::Cuts::cuts.begin(), i);
    const auto &key = it->first;
    auto &val = it->second;

    // Retrieve the Gurobi variable corresponding to the subject graph edge. If
    // the edge corresponds to a dataflow channel, then we need to retrieve the
    // variable corresponding to the pathIn of the channel.
    GRBVar nodeVar = retrieveAndSearchGRBVar(key, "pathIn", pathInSearcher);
    std::set<std::string> fanIns = anchorsRemoved.getFanins(key);

    if (fanIns.size() == 1) {
      // if a node has single fanin, then it is not mapped to LUTs. The delay of
      // the node is equal to the delay of the fanin.
      // No cut should be selected for these nodes, the other cuts are present
      // for the correct cut enumeration.
      GRBVar faninVar =
          retrieveAndSearchGRBVar(*fanIns.begin(), "pathOut", pathOutSearcher);
      // model.addConstr(nodeVar == faninVar, "single_fanin_delay");
      localConstraintsEqual.emplace_back(nodeVar, faninVar, 0);
      continue;
    }

    for (auto &cut : val) {
      // Loop over each cut of the subject graph edge
      GRBVar &cutSelectionVar = cut.cutSelection;
      if ((cut.leaves.size() == 1) && (*cut.leaves.begin() == key)) {
        // if the cut has a single leaf and it is equal to the root node, then
        // it's the trivial cut. The LUT is created by the fanins of the root
        // node.
        for (auto &fanIn : fanIns) {
          GRBVar faninVar =
              retrieveAndSearchGRBVar(fanIn, "pathOut", pathOutSearcher);
          // model.addConstr(nodeVar + (1 - cutSelectionVar) * bigConstant >=
          //                     faninVar + lutDelay,
          //                 "trivial_cut_delay");
          localConstraints.emplace_back(nodeVar +
                                            (1 - cutSelectionVar) * bigConstant,
                                        faninVar + lutDelay, 1);
        }
        continue;
      }

      for (auto &leaf : cut.leaves) {
        // Loop over each leaf of the cut

        // Retrieve the Gurobi variable corresponding to the leaf. If the leaf
        // is a dataflow channel, then retrieve the pathOut variable of the
        // channel.
        GRBVar leafVar =
            retrieveAndSearchGRBVar(leaf, "pathOut", pathOutSearcher);

        // Add the delay propagation constraint
        // model.addConstr(nodeVar + (1 - cutSelectionVar) * bigConstant >=
        //                     leafVar + lutDelay,
        //                 "delay_propagation");

        localConstraints.emplace_back(nodeVar +
                                          (1 - cutSelectionVar) * bigConstant,
                                      leafVar + lutDelay, 2);

        // Get the path from the leaf to the root
        std::vector<std::string> path;

#pragma omp critical
        {
          path = getOrCreateLeafToRootPath(key, leaf, leafToRootPaths,
                                           anchorsRemoved);
        }
        for (auto &nodePath : path) {
          // Loop over edges in the path from the leaf to the root. Add cut
          // selection conflict constraints for channels that are on the path.
          std::string nodeWithChannel =
              retrieveChannelName(nodePath, "buffer").str();
          auto bufferVars = bufSearcher.variableIncludes(nodeWithChannel);
          if (nodeWithChannel != nodePath && bufferVars) {
            // No buffer can be placed on a channel if the chosen cut covers
            // that channel.

            // model.addConstr(1 >= bufferVars.value() + cutSelectionVar,
            //                 "cut_selection_conflict");
            localConstraints.emplace_back(
                1, bufferVars.value() + cutSelectionVar, 3);
          }
        }
      }
    }
#pragma omp critical
    {
      // Gurobi does not allow to modify the model from multiple threads, so we need to add the constraints in a critical section
      constraints.splice(constraints.end(), localConstraints);
      constraintsEqual.splice(constraintsEqual.end(), localConstraintsEqual);
    }
  }

  model.update();

  for (auto &constraint : constraints) {
    model.addConstr(
        constraint.lhs >= constraint.rhs,
        static_cast<std::string>(getConstraintName(constraint.constraintType)));
  }

  for (auto &constraint : constraintsEqual) {
    model.addConstr(
        constraint.lhs == constraint.rhs,
        static_cast<std::string>(getConstraintName(constraint.constraintType)));
  }

  SmallVector<CFDFC *> cfdfcs;
  for (auto [cfdfc, optimize] : funcInfo.cfdfcs) {
    if (!optimize)
      continue;
    cfdfcs.push_back(cfdfc);
    addCFDFCVars(*cfdfc);
    addChannelThroughputConstraints(*cfdfc);
    addUnitThroughputConstraints(*cfdfc);
  }
  addObjective(allChannels, cfdfcs);


  llvm::errs() << "model marked ready to optimize"
               << "\n";
  markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
