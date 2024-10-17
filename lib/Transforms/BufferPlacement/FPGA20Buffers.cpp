//===- FPGA20Buffers.cpp - FPGA'20 buffer placement -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements FPGA'20 smart buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "gurobi_c.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpga20;

FPGA20Buffers::FPGA20Buffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, bool legacyPlacement)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod),
      legacyPlacement(legacyPlacement) {
  if (!unsatisfiable)
    setup();
}

FPGA20Buffers::FPGA20Buffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, bool legacyPlacement,
                             Logger &logger, StringRef milpName)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                          milpName),
      legacyPlacement(legacyPlacement) {
  if (!unsatisfiable)
    setup();
}

void FPGA20Buffers::extractResult(BufferPlacement &placement) {
  // Iterate over all channels in the circuit
  for (auto &[channel, channelVars] : vars.channelVars) {
    // Extract number and type of slots from the MILP solution, as well as
    // channel-specific buffering properties
    unsigned numSlotsToPlace = static_cast<unsigned>(
        channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    if (numSlotsToPlace == 0)
      continue;

    bool placeOpaque = channelVars.signalVars[SignalType::DATA].bufPresent.get(
                           GRB_DoubleAttr_X) > 0;

    handshake::ChannelBufProps &props = channelProps[channel];

    PlacementResult result;
    if (placeOpaque) {
      if (legacyPlacement) {
        // Satisfy the transparent slots requirement, all other slots are opaque
        result.numTrans = props.minTrans;
        result.numOpaque = numSlotsToPlace - props.minTrans;
      } else {
        // We want as many slots as possible to be transparent and at least one
        // opaque slot, while satisfying all buffering constraints
        unsigned actualMinOpaque = std::max(1U, props.minOpaque);
        if (props.maxTrans.has_value() &&
            (props.maxTrans.value() < numSlotsToPlace - actualMinOpaque)) {
          result.numTrans = props.maxTrans.value();
          result.numOpaque = numSlotsToPlace - result.numTrans;
        } else {
          result.numOpaque = actualMinOpaque;
          result.numTrans = numSlotsToPlace - result.numOpaque;
        }
      }
    } else {
      // All slots should be transparent
      result.numTrans = numSlotsToPlace;
    }

    result.deductInternalBuffers(Channel(channel), timingDB);
    placement[channel] = result;
  }

  if (logger)
    logResults(placement);
}

void FPGA20Buffers::addCustomChannelConstraints(Value channel) {
  ChannelVars &chVars = vars.channelVars[channel];
  handshake::ChannelBufProps &props = channelProps[channel];
  GRBVar &dataBuf = chVars.signalVars[SignalType::DATA].bufPresent;

  if (props.minOpaque > 0) {
    // Force the MILP to use opaque slots
    model.addConstr(dataBuf == 1, "custom_forceOpaque");
    if (props.minTrans > 0) {
      // If the properties ask for both opaque and transparent slots, let
      // opaque slots take over. Transparents slots will be placed "manually"
      // from the total number of slots indicated by the MILP's result
      unsigned minTotalSlots = props.minOpaque + props.minTrans;
      model.addConstr(chVars.bufNumSlots >= minTotalSlots,
                      "custom_minOpaqueAndTrans");
    } else {
      // Force trhe MILP to place a minimum number of opaque slots
      model.addConstr(chVars.bufNumSlots >= props.minOpaque,
                      "custom_minOpaque");
    }
  } else if (props.minTrans > 0) {
    // Force the MILP to place a minimum number of transparent slots
    model.addConstr(chVars.bufNumSlots >= props.minTrans + dataBuf,
                    "custom_minTrans");
  }
  if (props.minOpaque + props.minTrans > 0)
    model.addConstr(chVars.bufPresent == 1, "custom_forceBuffers");

  // Set a maximum number of slots to be placed
  if (props.maxOpaque.has_value()) {
    if (*props.maxOpaque == 0) {
      // Force the MILP to use transparent slots
      model.addConstr(dataBuf == 0, "custom_forceTransparent");
    }
    if (props.maxTrans.has_value()) {
      // Force the MILP to use a maximum number of slots
      unsigned maxSlots = *props.maxTrans + *props.maxOpaque;
      if (maxSlots == 0) {
        model.addConstr(chVars.bufPresent == 0, "custom_noBuffers");
        model.addConstr(chVars.bufNumSlots == 0, "custom_noSlots");
      } else {
        model.addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
      }
    }
  }
}

std::optional<GRBVar> variableExistss(GRBModel &model,
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

void FPGA20Buffers::setup() {
  // // Count the number of operations in the function
  // int numOps = 0;
  // for (Operation &op : funcInfo.funcOp.getOps()) {
  //   ++numOps;
  // }

  // std::vector<Value> allChannels;
  // std::vector<std::vector<GRBVar>> varVector;
  // for (auto &[channel, _] : channelProps) {
  //       if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
  //       !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())){
  //   allChannels.push_back(channel);
  //   ChannelVars &channelVars = vars.channelVars[channel];
  //   std::string suffix = "_" + getUniqueName(*channel.getUses().begin());
  //   auto createVar = [&](const llvm::Twine &name, char type) {
  //     return model.addVar(0, 2, 0.0, type, (name + suffix).str());
  //   };
  //   ChannelSignalVars &signalVars = channelVars.signalVars[SignalType::DATA];
  //   signalVars.bufPresent = createVar("dataBufPresent", GRB_BINARY);

  //   channelVars.bufPresent = createVar("bufPresent", GRB_BINARY);
  //   channelVars.bufNumSlots = createVar("bufNumSlots", GRB_INTEGER);

  //   model.update();

  //   model.addConstr(channelVars.bufPresent == signalVars.bufPresent,
  //                   "buf_slots_positive");
  //   model.addConstr(channelVars.bufNumSlots == signalVars.bufPresent * 2,
  //                   "buf_slots_positive");

  //   GRBVar before;
  //   Operation *defOp = channel.getDefiningOp();
  //   if (defOp) {
  //     std::string uniqueOpName =
  //         defOp->getName().getStringRef().str() + "_" +
  //         std::to_string(reinterpret_cast<uintptr_t>(defOp));
  //     // Check if a variable with the same name already exists in the model
  //     std::optional<GRBVar> beforeVar = variableExistss(model, uniqueOpName);
  //     if (beforeVar.has_value()) {
  //       before = beforeVar.value();
  //     } else {
  //       before = model.addVar(0, numOps - 1, 0.0, GRB_INTEGER, uniqueOpName);
  //     }
  //   } else {
  //     before = model.addVar(0, 0, 0.0, GRB_INTEGER, "before");
  //   }

  //   GRBVar after;
  //   if (!channel.use_empty()) {
  //     Operation *useOp = *channel.user_begin();
  //     std::string uniqueOpName =
  //         useOp->getName().getStringRef().str() + "_" +
  //         std::to_string(reinterpret_cast<uintptr_t>(useOp));
  //     std::optional<GRBVar> afterVar = variableExistss(model, uniqueOpName);
  //     if (afterVar.has_value()) {
  //       after = afterVar.value();
  //     } else {
  //       after = model.addVar(0, numOps - 1, 0.0, GRB_INTEGER, uniqueOpName);
  //     }
  //   } else {
  //     after = model.addVar(0, 0, 0.0, GRB_INTEGER, "after");
  //   }

  //   varVector.push_back({signalVars.bufPresent, before, after});

  //   model.update();
  // }
  // }

  // GRBLinExpr obj = 0;
  // for (const auto &entry : varVector) {
  //   obj += entry[0];
  // }

  // model.setObjective(obj, GRB_MINIMIZE);
  // model.update();

  // for (auto &entry : varVector) {
  //   GRBVar &bufVar = entry[0];
  //   GRBVar &before = entry[1];
  //   GRBVar &after = entry[2];
  //   model.addConstr(after - before + 100 * bufVar >= 1, "buf_order");
  // }

  // model.update();

  // Signals for which we have variables
  SmallVector<SignalType, 1> signals;
  signals.push_back(SignalType::DATA);

  // /// NOTE: (lucas-rami) For each buffering group this should be the timing
  // /// model of the buffer that will be inserted by the MILP for this group.
  // We
  // /// don't have models for these buffers at the moment therefore we provide
  // a
  /// null-model to each group, but this hurts our placement's accuracy.
  const TimingModel *bufModel = nullptr;

  // Create buffering groups. In this MILP we only care for the data signal
  SmallVector<BufferingGroup> bufGroups;
  bufGroups.emplace_back(ArrayRef<SignalType>{SignalType::DATA}, bufModel);

  // Create channel variables and constraints
  std::vector<Value> allChannels;
  for (auto &[channel, _] : channelProps) {
    allChannels.push_back(channel);
    addChannelVars(channel, signals);
    //addCustomChannelConstraints(channel);

        ChannelVars &channelVars = vars.channelVars[channel];
    std::string suffix = "_" + getUniqueName(*channel.getUses().begin());
    auto createVar = [&](const llvm::Twine &name, char type) {
      return model.addVar(0, 1, 0.0, type, (name + suffix).str());
    };
    ChannelSignalVars &signalVars = channelVars.signalVars[SignalType::DATA];
    signalVars.bufPresent = createVar("dataBufPresent", GRB_BINARY);

    channelVars.bufPresent = createVar("bufPresent", GRB_BINARY);
    channelVars.bufNumSlots = createVar("bufNumSlots", GRB_INTEGER);

    model.update();

    model.addConstr(channelVars.bufPresent == signalVars.bufPresent,
                    "buf_slots_positive");
    model.addConstr(channelVars.bufNumSlots == signalVars.bufPresent,
                    "buf_slots_positive");

    // Add path and elasticity constraints over all channels in the function
    // that are not adjacent to a memory interface
    if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
        !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
      //addChannelPathConstraints(channel, SignalType::DATA, bufModel);
      addChannelElasticityConstraints(channel, bufGroups);
    }
  }

  // Add path and elasticity constraints over all units in the function
  for (Operation &op : funcInfo.funcOp.getOps()) {
    //addUnitPathConstraints(&op, SignalType::DATA);
    addUnitElasticityConstraints(&op);
  }

  GRBLinExpr obj = 0;
  for (auto &[channel, channelVars] : vars.channelVars) {
    obj += channelVars.signalVars[SignalType::DATA].bufPresent;
  }

  model.setObjective(obj, GRB_MINIMIZE);

  // // Create CFDFC variables and add throughput constraints for each CFDFC
  // // that
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
  markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
