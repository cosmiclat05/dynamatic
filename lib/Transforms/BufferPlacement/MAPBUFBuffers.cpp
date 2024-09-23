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
#include "gurobi_c.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <omp.h>
#include <list>
#include <string>

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

std::optional<GRBVar> variableIncludes(const std::string &subStr, const std::vector<GRBVar>& channelVarsVec) {
  int numVars = channelVarsVec.size();
  // Loop through all variables and check their names
  for (int i = 0; i < numVars; i++) {
    std::string varName = channelVarsVec[i].get(GRB_StringAttr_VarName);
    if (varName.find(subStr) != std::string::npos)
      return channelVarsVec[i]; // Variable exists
  }
  return {}; // Variable does not exist
}

std::vector<GRBVar> variableVectors(const std::string &subStr, const std::vector<GRBVar>& channelVarsVec) {
  int numVars = channelVarsVec.size();
  std::vector<GRBVar> variables;
  // Loop through all variables and check their names
  for (int i = 0; i < numVars; i++) {
    std::string varName = channelVarsVec[i].get(GRB_StringAttr_VarName);
    if (varName.find(subStr) != std::string::npos)
      variables.push_back(channelVarsVec[i]); 
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


enum ConstraintNames
{
    trivialCut_noBuf, //1
    trivialCut_Buf, //2
    oneCut_noBuf,  //3
    oneCut_Buf, //4
    moreCut_noBuf, //5
    moreCut_Buf, //6
    cut_selection_conflict, //7
    equality //8

};

constexpr std::string_view getConstraintName(ConstraintNames constraint)
{
    switch (constraint)
    {
      case oneCut_noBuf:             return "oneCut_noBuf";
      case oneCut_Buf:               return "oneCut_Buf";
      case moreCut_noBuf:            return "moreCut_noBuf";
      case moreCut_Buf:              return "moreCut_Buf";
      case trivialCut_noBuf:         return "trivialCut_noBuf";
      case trivialCut_Buf:           return "trivialCut_Buf";
      case cut_selection_conflict:   return "cut_selection_conflict";
      case equality:                 return "equality";
    }
}

struct GrbConst {
  GRBLinExpr lhs;
  GRBLinExpr rhs;
  ConstraintNames constraintType; 
  GrbConst(const GRBLinExpr &lhs, const GRBLinExpr &rhs, int type) : lhs(lhs), rhs(rhs), constraintType(static_cast<ConstraintNames>(type)) {}
};

class VariableSearcher {
private:
    std::unordered_map<std::string, std::vector<GRBVar>> variableMap;

public:
    VariableSearcher(const std::vector<GRBVar>& channelVarsVec) {
        for (const auto& var : channelVarsVec) {
            std::string varName = var.get(GRB_StringAttr_VarName);
            variableMap[varName].push_back(var);
        }
    }

    std::optional<GRBVar> variableIncludes(const std::string& subStr) const {
        for (const auto& pair : variableMap) {
            if (pair.first.find(subStr) != std::string::npos) {
                return pair.second.front();
            }
        }
        return {};
    }

    std::vector<GRBVar> variableVectors(const std::string& subStr) const {
        std::vector<GRBVar> result;
        for (const auto& pair : variableMap) {
            if (pair.first.find(subStr) != std::string::npos) {
                result.insert(result.end(), pair.second.begin(), pair.second.end());
            }
        }
        return result;
    }
};

bool isChannelVar(const std::string& node){
  return node.find("new") == std::string::npos && node.find('.') == std::string::npos && node.find('_') != std::string::npos;
}


std::string retrieveChannelName(const std::string& node, const std::string& variableType) {
  if (!isChannelVar(node)){
    return {};
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
    bufferVarNameStream << (result.back() == "valid" ? ("valid" + variableTypeName) : ("ready" + variableTypeName))
                        << leafNodeNameTillUnderScore;
  } 
  else {
    const auto leafLastPar = node.find_last_of('[');
    const auto leafNodeNameTillPar = node.substr(0, leafLastPar);
    bufferVarNameStream << ("data" + variableTypeName) << leafNodeNameTillPar;
  }

  return bufferVarNameStream.str();
}

void MAPBUFBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signals;
  signals.push_back(SignalType::DATA);
  signals.push_back(SignalType::VALID);
  signals.push_back(SignalType::READY);

  experimental::BlifParser parser;
  experimental::BlifData blifFile = parser.parseBlifFile("/home/oyasar/mapbuf_external/removal_blifs/unanchored_small.blif");

  experimental::Cuts cutfile;
  cutfile.readFromFile("/home/oyasar/mapbuf_external/cuts/cuts_small.txt");

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
  
  GRBVar clock = model.addVar(0, 11.9, 0.0, GRB_CONTINUOUS, "clock");
  model.update();
  // Create channel variables and constraints
  
  std::vector<Value> allChannels;
  std::vector<GRBVar> bufVarsVector; 
  std::vector<GRBVar> pathInVarsVector; 
  std::vector<GRBVar> pathOutVarsVector;  
  for (auto &[channel, _] : channelProps) {
    allChannels.push_back(channel);
    addChannelVars(channel, signals);
    addCustomChannelConstraints(channel);

    for(auto& signal : signals){
      ChannelVars &channelVars = vars.channelVars[channel];
      ChannelSignalVars &signalVars = channelVars.signalVars[signal];
      GRBVar& bufVarSignal = vars.channelVars[channel].signalVars[signal].bufPresent;
      GRBVar &t1 = signalVars.path.tIn;
      GRBVar &t2 = signalVars.path.tOut;
      model.addConstr(t2 <= clock, "pathOut_period");
      model.addConstr(t1 <= clock, "pathIn_period");
      model.addConstr(t2 - t1 + 100 * bufVarSignal >= 0, "buf_delay");

      pathInVarsVector.push_back(t1);
      pathOutVarsVector.push_back(t2);
      bufVarsVector.push_back(bufVarSignal);
    }

    ChannelVars &channelVars = vars.channelVars[channel];
    //GRBVar &tIn = channelVars.elastic.tIn;
    //GRBVar &tOut = channelVars.elastic.tOut;
    GRBVar &bufPresent = channelVars.bufPresent;
    GRBVar &bufNumSlots = channelVars.bufNumSlots;

    // If there is at least one slot, there must be a buffer
    model.addConstr(0.01 * bufNumSlots <= bufPresent, "elastic_presence");

    // Compute the sum of the binary buffer presence over all signals that have
    // different buffers
    GRBLinExpr disjointBufPresentSum;
    for (const BufferingGroup &group : bufGroups) {
      GRBVar &groupBufPresent =
          channelVars.signalVars[group.getRefSignal()].bufPresent;
      disjointBufPresentSum += groupBufPresent;

      // For each group, the binary buffer presence variable of different signals
      // must be equal
      StringRef refName = getSignalName(group.getRefSignal());
      for (SignalType sig : group.getOtherSignals()) {
        StringRef otherName = getSignalName(sig);
        model.addConstr(groupBufPresent == channelVars.signalVars[sig].bufPresent,
                        "elastic_" + refName.str() + "_same_" + otherName.str());
      }
    }

    // There must be enough slots for all disjoint buffers
    model.addConstr(disjointBufPresentSum <= bufNumSlots, "elastic_slots");

    for (auto &[sig, signalVars] : channelVars.signalVars) {
      // If there is a buffer present on a signal, then there is a buffer present
      // on the channel
      model.addConstr(signalVars.bufPresent <= bufPresent,
                      "elastic_" + getSignalName(sig).str() + "Presence");
    }
  }

  std::unordered_map<std::string, GRBVar> nodeToGRB;
  std::unordered_map<std::string, GRBVar> channelToGRB;
  std::vector<GRBVar> relaxationVars;
  
  for (auto& node : blifFile.getNodes()){
    GRBVar& nodeVar = nodeToGRB[node];
    nodeVar = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, node);
    GRBVar clockRelax = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "clockRelax_" + node);
    relaxationVars.push_back(clockRelax);
    model.update();
    if (blifFile.isPrimaryInput(node)){
      model.addConstr(nodeVar == 0, "input_delay");
    }
    else
    {
      model.addConstr(nodeVar <= clock + clockRelax, "clock_period_constraint");
    }
  }
  model.update();

  for (auto &[key, val] : cutfile.cuts) {
    GRBLinExpr cutSelectionSum = 0;
    int i = 0;
    for (auto &cut : val) {
      GRBVar &cutSelection = cut.cutSelection;      
      cutSelection = model.addVar(0, GRB_INFINITY, 0, GRB_BINARY, (cut.node + "__CutSelection_" + std::to_string(i)));
      cutSelectionSum += cutSelection;
      i++;
    }
    model.update();
    model.addConstr(cutSelectionSum == 1, "cut_selection_constraint");
  }

  VariableSearcher bufSearcher(bufVarsVector);
  VariableSearcher pathInSearcher(pathInVarsVector);
  VariableSearcher pathOutSearcher(pathOutVarsVector);
  llvm::errs() << "added all cuts" << "\n";

  std::list<GrbConst> constraints;
  std::list<GrbConst> constraintsEqual;
  int n = cutfile.cuts.size();
  
  omp_set_num_threads(4);
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < n; ++i) {
    std::list<GrbConst> localConstraints;
    auto it = std::next(cutfile.cuts.begin(), i);
    const auto& key = it->first;
    auto& val = it->second;

    GRBVar& nodeVar = nodeToGRB[key];
    int numCuts = val.size();
    bool bufferVar = true;
    GRBVar bufferVarGRB;

    std::string bufVariableName = retrieveChannelName(key, "buffer");
    if (bufVariableName.empty()){
      bufferVar = false;
    }
    else 
    {
      std::string pathInName = retrieveChannelName(key, "pathIn");
      auto inVarOption = pathInSearcher.variableIncludes(pathInName);
      if (inVarOption) {
        //llvm::errs() << bufferVarName << "\n";
        nodeVar = inVarOption.value();
      } 
      else {
        //llvm::errs() << "\nno gurobi pathIn for " + pathInName;  
      }

      auto bufferVarOption = bufSearcher.variableIncludes(bufVariableName);
      if (bufferVarOption) {
        //llvm::errs() << bufferVarName << "\n";
        bufferVarGRB = bufferVarOption.value();
      } 
      else {
        bufferVar = false;
        //llvm::errs() << "\nno gurobi buffer for " + bufVariableName;  
      }
    }

    for (auto &cut : val){
      GRBVar &cutSelectionVar = cut.cutSelection;
      if ((cut.leaves.size() == 1) && (cut.leaves.at(0) == key)){  //trivial cut
        std::set<std::string> fanIns = blifFile.getNodeFanins(key);

        for (auto &fanIn : fanIns){
          GRBVar& faninVar = nodeToGRB[fanIn];
          if (fanIns.size() == 1){
            localConstraints.emplace_back(nodeVar, faninVar, 7);
            break;
          }
          if (!bufferVar){
            //model.addConstr(nodeVar + (1 - cutSelectionVar) * 100 >= faninVar + 0.7, "trivialCut_noBuf");
            localConstraints.emplace_back(nodeVar + (1 - cutSelectionVar) * 100, faninVar + 0.7, 0);
          }
          else if (bufferVar){
            //model.addConstr(nodeVar + (1 - cutSelectionVar + bufferVarGRB) * 100 >= faninVar + 0.7, "trivialCut_Buf");
            localConstraints.emplace_back(nodeVar + (1 - cutSelectionVar + bufferVarGRB) * 100, faninVar + 0.7, 1);
          }
        }
        continue;
      }

      for (auto &leaf : cut.leaves){
        GRBVar& leafVar = nodeToGRB[leaf];
        std::string pathOutName = retrieveChannelName(leaf, "pathOut");

        if (!pathOutName.empty()){
          auto outVarOption = pathInSearcher.variableIncludes(pathOutName);
          outVarOption = pathOutSearcher.variableIncludes(pathOutName);
          if (outVarOption) {
            //llvm::errs() << bufferVarName << "\n";
            leafVar = outVarOption.value();
          } 
          else {
            //llvm::errs() << "\nno gurobi pathOut for " + pathOutName;  
          }
        }

        if ((numCuts == 1) && (!bufferVar)){
          //model.addConstr(nodeVar >= leafVar + 0.7, "oneCut_noBuf");
          localConstraints.emplace_back(nodeVar, leafVar + 0.7, 2);
        }
        else if ((numCuts == 1) && (bufferVar)){
          // model.addConstr(nodeVar + bufferVarGRB * 100 >= leafVar + 0.7, "oneCut_Buf");
          localConstraints.emplace_back(nodeVar + bufferVarGRB * 100, leafVar + 0.7, 3);
        }
        else if ((numCuts > 1) && (!bufferVar)){
          //model.addConstr(nodeVar + (1 - cutSelectionVar) * 100 >= leafVar + 0.7, "moreCut_noBuf");
          localConstraints.emplace_back(nodeVar + (1 - cutSelectionVar) * 100, leafVar + 0.7, 4);
        }
        else if ((numCuts > 1) && (bufferVar)){
          //model.addConstr(nodeVar + (1 - cutSelectionVar + bufferVarGRB) * 100 >= leafVar + 0.7, "moreCut_Buf");
          localConstraints.emplace_back(nodeVar + (1 - cutSelectionVar + bufferVarGRB) * 100, leafVar + 0.7, 5);
        }
        
        std::vector<std::string> path = blifFile.findPath(leaf, key);
        std::vector<std::string> nodesWithChannels;
        
        for (auto& nodePath : path){
          std::string nodeWithChannel = retrieveChannelName(nodePath, "buffer");

          if (!nodeWithChannel.empty()){
            auto bufferVars = bufSearcher.variableIncludes(nodeWithChannel);
            GRBVar bufferLeaf;
            if (bufferVars) {
              //llvm::errs() << bufferVarName << "\n";
              bufferLeaf = bufferVars.value();
              localConstraints.emplace_back(1, bufferLeaf + cutSelectionVar, 6);
            } 
          }
        }
      }
    }
    #pragma omp critical
    {
      constraints.splice(constraints.end(), localConstraints);
    }
  }
  
  llvm::errs() << "model description finished" << "\n";

  for (auto& constraint : constraints){
    model.addConstr(constraint.lhs >= constraint.rhs, static_cast<std::string>(getConstraintName(constraint.constraintType)));
  }

  llvm::errs() << "constraints are written" << "\n";

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

  // // unsigned totalExecs = 0;
  // // for (Value channel : allChannels) {
  // //   totalExecs += getChannelNumExecs(channel);
  // // }

  // // // Create the expression for the MILP objective
  // // GRBLinExpr objective;

  // // // For each CFDFC, add a throughput contribution to the objective, weighted
  // // // by the "importance" of the CFDFC
  // // double maxCoefCFDFC = 0.0;
  // // double fTotalExecs = static_cast<double>(totalExecs);
  // // if (totalExecs != 0) {
  // //   for (CFDFC *cfdfc : cfdfcs) {
  // //     double coef = (cfdfc->channels.size() * cfdfc->numExecs) / fTotalExecs;
  // //     objective += coef * vars.cfVars[cfdfc].throughput;
  // //     maxCoefCFDFC = std::max(coef, maxCoefCFDFC);
  // //   }
  // // }

  // // // In case we ran the MILP without providing any CFDFC, set the maximum CFDFC
  // // // coefficient to any positive value
  // // if (maxCoefCFDFC == 0.0)
  // //   maxCoefCFDFC = 1.0;

  // // // For each channel, add a "penalty" in case a buffer is added to the channel,
  // // // and another penalty that depends on the number of slots
  // // double bufPenaltyMul = 1e-4;
  // // double slotPenaltyMul = 1e-5;
  // // for (Value channel : allChannels) {
  // //   ChannelVars &channelVars = vars.channelVars[channel];
  // //   objective -= maxCoefCFDFC * bufPenaltyMul * channelVars.bufPresent;
  // //   objective -= maxCoefCFDFC * slotPenaltyMul * channelVars.bufNumSlots;
  // // }

  //addObjective(allChannels, cfdfcs);

  GRBLinExpr objective = clock;
    for (auto& relax : relaxationVars){
    objective += 0.1 * relax;
  }

  // double bufPenaltyMul = 1e-4;
  // double slotPenaltyMul = 1e-5;
  // for (Value channel : allChannels) {
  //   ChannelVars &channelVars = vars.channelVars[channel];
  //   objective += 1 * bufPenaltyMul * channelVars.bufPresent;
  //   objective += 1 * slotPenaltyMul * channelVars.bufNumSlots;
  // }
  
  model.setObjective(objective, GRB_MINIMIZE);

  //model.set(GRB_IntParam_DualReductions, 0);
  //model.feasRelax(GRB_FEASRELAX_LINEAR, false, false, true);
  
  llvm::errs() << "model marked ready to optimize" << "\n";
  markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
