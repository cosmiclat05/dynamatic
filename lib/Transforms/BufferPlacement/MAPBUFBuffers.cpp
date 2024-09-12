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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "experimental/Support/CutEnumeration.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::mapbuf;

MAPBUFBuffers::MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod) {
  if (!unsatisfiable)
    setup();
}

MAPBUFBuffers::MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, Logger &logger,
                             StringRef milpName)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                          milpName) {
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

std::optional<GRBVar> variableIncludes(GRBModel &model,
                                       const std::string &subStr) {
  GRBVar *vars = model.getVars();
  int numVars = model.get(GRB_IntAttr_NumVars);
  // Loop through all variables and check their names
  for (int i = 0; i < numVars; i++) {
    std::string varName = vars[i].get(GRB_StringAttr_VarName);
    if (varName.find(subStr) != std::string::npos)
      return vars[i]; // Variable exists
  }
  return {}; // Variable does not exist
}

std::vector<GRBVar> variableVectors(GRBModel &model,
                                       const std::string &subStr) {
  GRBVar *vars = model.getVars();
  int numVars = model.get(GRB_IntAttr_NumVars);
  std::vector<GRBVar> variables;
  // Loop through all variables and check their names
  for (int i = 0; i < numVars; i++) {
    std::string varName = vars[i].get(GRB_StringAttr_VarName);
    if (varName.find(subStr) != std::string::npos)
      variables.push_back(vars[i]); 
  }
  return variables;
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

void MAPBUFBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signals;
  signals.push_back(SignalType::DATA);
  signals.push_back(SignalType::VALID);
  signals.push_back(SignalType::READY);

  experimental::BlifParser parser;
  experimental::BlifData blifFile = parser.parseBlifFile("/home/oyasar/mapbuf_external/removal_blifs/unanchored_blif.blif");

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

  // Create channel variables and constraints
  std::vector<Value> allChannels;

  experimental::Cuts cutfile;
  cutfile.readFromFile("/home/oyasar/mapbuf_external/cuts.txt");

  for (auto &[channel, _] : channelProps) {
    allChannels.push_back(channel);
    addChannelVars(channel, signals);
    addCustomChannelConstraints(channel);

    ChannelVars &channelVars = vars.channelVars[channel];
    GRBVar &tIn = channelVars.elastic.tIn;
    GRBVar &tOut = channelVars.elastic.tOut;
    GRBVar &bufPresent = channelVars.bufPresent;
    GRBVar &bufNumSlots = channelVars.bufNumSlots;

    // If there is at least one slot, there must be a buffer
    model.addConstr(0.01 * bufNumSlots <= bufPresent, "elastic_presence");

    for (auto &[sig, signalVars] : channelVars.signalVars) {
      // If there is a buffer present on a signal, then there is a buffer present
      // on the channel
      model.addConstr(signalVars.bufPresent <= bufPresent,
                      "elastic_" + getSignalName(sig).str() + "Presence");
    }
  }

  for (auto& nodes : blifFile.getNodesInOrder()){
    GRBVar nodeVar = model.addVar(0, targetPeriod, 0.0, GRB_CONTINUOUS, nodes);
  }

  model.update();

  for (auto& input : blifFile.getPrimaryInputs()){
    GRBVar nodeInput = model.getVarByName(input);
    model.addConstr(nodeInput == 0, "input_delay");
  }

  for (auto &[key, val] : cutfile.cuts) {
    GRBLinExpr cutSelectionSum = 0;
    int i = 0;
    for (auto &cut : val) {
      // GRBVar &cutNodeVar = cut.nodeVar;
      // cutNodeVar = nodeVar;
      GRBVar &cutSelection = cut.cutSelection;      
      cutSelection =
          model.addVar(0, 1, 0, GRB_BINARY,
                       (cut.node + "__CutSelection_" + std::to_string(i)));
      cutSelectionSum += cutSelection;
      i++;
    }
    model.update();
    GRBVar nodeVar = model.getVarByName(key);
    model.addConstr(nodeVar <= targetPeriod, "clock_period_constraint");
    model.addConstr(0 <= nodeVar, "clock_period_constraint_zero");
    model.addConstr(cutSelectionSum == 1, "cut_selection_constraint");
  }

  std::string bufferVarName;
  std::size_t lastUnderscore;
  std::string nodeNameTillUnderScore;
  std::string channelTypeName;

  for (auto const &[key, val] : cutfile.cuts) {
    GRBVar nodeVar = model.getVarByName(key);
    int i = 0;
    int numCuts = val.size();
    int bufferVar = 1;

    lastUnderscore = key.find_last_of('_');
    nodeNameTillUnderScore = key.substr(0, lastUnderscore);
    GRBVar bufferVarGRB;

    if ((key.find("new") != std::string::npos) || (lastUnderscore == std::string::npos)){
      bufferVar = 0;
    }
    else {
      channelTypeName = key.substr(lastUnderscore + 1);

      if (channelTypeName.find("valid") != std::string::npos)
      {
        bufferVarName = "validBufPresent_" + nodeNameTillUnderScore;
      }
      else if (channelTypeName.find("ready") != std::string::npos)
      {
        bufferVarName = "readyBufPresent_" + nodeNameTillUnderScore;
      }
      else
      {
        std::size_t lastParanthesis = key.find_last_of('[');
        std::string nameTillPar = key.substr(0, lastParanthesis);
        bufferVarName = "dataBufPresent_" + nameTillPar;
      }
      
      std::optional<GRBVar> bufferVarOption = variableIncludes(model, bufferVarName);
      if (bufferVarOption) {
        //llvm::errs() << bufferVarName << "\n";
        bufferVarGRB = bufferVarOption.value();
      } else {
        bufferVar = 0;
        //llvm::errs() << "\nno gurobi buffer for " + bufferVarName;  
      }
    }
    
    for (auto &cut : val){
      GRBVar cutSelectionVar = model.getVarByName(cut.node + "__CutSelection_" + std::to_string(i));
      i++;
      for (auto &leaf : cut.leaves){
        // std::optional<GRBVar> leafVarOption = variableExists(model, leaf);
        GRBVar leafVar = model.getVarByName(leaf);

        // if (leafVarOption) {
        //     leafVar = leafVarOption.value();
        // }
        // else { 
        //   //llvm::errs() << "No leafvar " + leaf << "\n";
        //   leafVar = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, leaf);
        //   model.update();
        //   model.addConstr(leafVar <= targetPeriod, "clock_period_constraint_leaf");
        //   model.addConstr(0 <= leafVar , "clock_period_constraint_leaf_zero");
        // }

        if ((numCuts == 1) && (bufferVar == 0)){
          model.addConstr(nodeVar >= leafVar + 1, "oneCut_noBuf");
        }
        else if ((numCuts == 1) && (bufferVar == 1)){
          model.addConstr(nodeVar + bufferVarGRB * 100 >= leafVar + 1, "oneCut_Buf");
        }
        else if ((numCuts > 1) && (bufferVar == 0)){
          model.addConstr(nodeVar + (1 - cutSelectionVar) * 100 >= leafVar + 1, "moreCut_noBuf");
        }
        else if ((numCuts > 1) && (bufferVar == 1)){
          model.addConstr(nodeVar + (1 - cutSelectionVar + bufferVarGRB) * 100 >= leafVar + 1, "moreCut_Buf");
        }

        lastUnderscore = leaf.find_last_of('_');
        nodeNameTillUnderScore = leaf.substr(0, lastUnderscore);
        channelTypeName = leaf.substr(lastUnderscore + 1);
        GRBVar bufferVarGRBleaf;

        std::vector<std::string> result;
        std::stringstream ss(leaf);
        std::string token;

        while (std::getline(ss, token, '_')) {
            result.push_back(token);
        }

        if (result.back() == "valid")
        {
          if (result.size() < 4) //buffer0_outs_ready
            continue;

          lastUnderscore = nodeNameTillUnderScore.find_last_of('_');
          nodeNameTillUnderScore = nodeNameTillUnderScore.substr(0, lastUnderscore);

          bufferVarName = "validBufPresent_" + nodeNameTillUnderScore;
        }
        else if (result.back() == "ready")
        {
          if (result.size() < 4) //buffer0_outs_ready
            continue;

          lastUnderscore = nodeNameTillUnderScore.find_last_of('_');
          nodeNameTillUnderScore = nodeNameTillUnderScore.substr(0, lastUnderscore);

          bufferVarName = "readyBufPresent_" + nodeNameTillUnderScore;
        }
        else
        {
          bufferVarName = "dataBufPresent_" + nodeNameTillUnderScore;
        }
          
        std::vector<GRBVar> bufferVars = variableVectors(model, bufferVarName);
        GRBLinExpr equalityConstraint;
        
        for (auto const& leafBufferVar : bufferVars)
        {
          equalityConstraint += leafBufferVar;
          model.addConstr(leafBufferVar + cutSelectionVar <= 1, "cut_selection_conflict");
        }

        if (bufferVars.size() > 1) {
          for (int i = 1; i < bufferVars.size(); i++)
          {
              model.addConstr(bufferVars[i] == bufferVars[0], "equality_constraint");
          }
        }
      }
    }
  }

  model.update();

  // // for all nodes, for all cuts of nodes, for all leaves of cuts, add
  // // constraints
  // for (auto const &[key, val] : cutfile.cuts) {
  //   int i = 0;
  //   for (auto cut : val) {
  //     GRBVar &nodeVar = cut.nodeVar;
  //     GRBVar cutSelectionVar = model.getVarByName(cut.node + "__CutSelection_" + std::to_string(i));
  //     i++;
  //     for (const auto &leaf : cut.leaves) {
  //       std::optional<GRBVar> leafVarOption = variableExists(model, leaf);
  //       if (leafVarOption) {
  //         GRBVar leafVar = leafVarOption.value();

  //         std::string leafName = leaf.substr(0, leaf.find('_'));
  //         std::string bufferVarName = "bufPresent_" + leafName;
  //         std::optional<GRBVar> bufferVarOption =
  //             variableIncludes(model, bufferVarName);
  //         if (bufferVarOption) {
  //           GRBVar bufferVar = bufferVarOption.value();
  //           model.addConstr(nodeVar + 1 <=
  //                               leafVar + (1 - cutSelectionVar + bufferVar) *
  //                                             targetPeriod,
  //                           "cut_constraint_channels");
  //           model.addConstr(bufferVar + cutSelectionVar <= 1,
  //                           "cut_selection_conflict");
  //         } else {
  //           model.addConstr(nodeVar + 1 <=
  //                               leafVar + (1 - cutSelectionVar) * targetPeriod,
  //                           "cut_constraint_no_channels");
  //         }
  //       }
  //     }
  //   }
  // }

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
  model.set(GRB_IntParam_DualReductions, 0);
  model.feasRelax(GRB_FEASRELAX_LINEAR, true, true, false);
  markReadyToOptimize();

  // for (auto &[channel, _] : channelProps) {
  //   allChannels.push_back(channel);
  //   addChannelVars(channel, signals);
  //   addCustomChannelConstraints(channel);

  //   // Add path and elasticity constraints over all channels in the function
  //   // that are not adjacent to a memory interface
  //   if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
  //       !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {

  //     addChannelPathConstraints(channel, SignalType::DATA, bufModel);
  //     addChannelElasticityConstraints(channel, bufGroups);
  //   }
  // }

  // // Add path and elasticity constraints over all units in the function
  // for (Operation &op : funcInfo.funcOp.getOps()) {
  //   addUnitPathConstraints(&op, SignalType::DATA);
  //   addUnitElasticityConstraints(&op);
  // }

  // // Create CFDFC variables and add throughput constraints for each CFDFC
  // that
  // // was marked to be optimized
  // SmallVector<CFDFC *> cfdfcs;
  // for (auto [cfdfc, optimize] : funcInfo.cfdfcs) {
  //   if (!optimize)
  //     continue;
  //   cfdfcs.push_back(cfdfc);
  //   addCFDFCVars(*cfdfc);
  //   addChannelThroughputConstraints(*cfdfc);
  //   addUnitThroughputConstraints(*cfdfc);
  // }

  // // Add the MILP objective and mark the MILP ready to be optimized
  // addObjective(allChannels, cfdfcs);
  // markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
