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

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::mapbuf;

MAPBUFBuffers::MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, experimental::Cuts cuts)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod) {
  if (!unsatisfiable)
    setup();
}

MAPBUFBuffers::MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo,
                             const TimingDatabase &timingDB,
                             double targetPeriod, experimental::Cuts cuts,
                             Logger &logger, StringRef milpName)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                          milpName) {
  if (!unsatisfiable)
    setup();
}

void MAPBUFBuffers::extractResult(BufferPlacement &placement) {
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

void MAPBUFBuffers::addMapbufConstraints(Value channel) {}

void MAPBUFBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signals;
  signals.push_back(SignalType::DATA);
  signals.push_back(SignalType::VALID);
  signals.push_back(SignalType::READY);

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

  for (auto &[channel, _] : channelProps) {
    allChannels.push_back(channel);
    ChannelVars &channelVars = vars.channelVars[channel];
    addChannelVars(channel, signals);
    for (auto signal : signals) {
      ChannelSignalVars &signalVars = channelVars.signalVars[signal];
      model.addConstr(signalVars.path.tOut <= targetPeriod, "path_period");
      model.addConstr(signalVars.path.tIn <= targetPeriod,"path_bufferedChannelIn");
      model.addConstr(signalVars.path.tOut - signalVars.path.tIn + targetPeriod * signalVars.bufPresent <= targetPeriod, "path_bufferedChannelIn");
    }
  }
      // Create the expression for the MILP objective
  GRBLinExpr objective;

  // For each channel, add a "penalty" in case a buffer is added to the channel,
  // and another penalty that depends on the number of slots
  double bufPenaltyMul = 1e-4;
  double slotPenaltyMul = 1e-5;
  for (Value channel : allChannels) {
    ChannelVars &channelVars = vars.channelVars[channel];
    objective -= bufPenaltyMul * channelVars.bufPresent;
    objective -= slotPenaltyMul * channelVars.bufNumSlots;
  }

  // Finally, set the MILP objective
  model.setObjective(objective, GRB_MAXIMIZE);
  markReadyToOptimize();
  /*
    for (auto &[channel, _] : channelProps) {
      allChannels.push_back(channel);
      addChannelVars(channel, signals);
      // addCustomChannelConstraints(channel);

      // Add path and elasticity constraints over all channels in the function
      // that are not adjacent to a memory interface
      if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
          !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {

        // addChannelPathConstraints(channel, SignalType::DATA, bufModel);
        // addChannelElasticityConstraints(channel, bufGroups);
      }
    }

    // Add path and elasticity constraints over all units in the function
    for (Operation &op : funcInfo.funcOp.getOps()) {
      addUnitPathConstraints(&op, SignalType::DATA);
      addUnitElasticityConstraints(&op);
    }

    // Create CFDFC variables and add throughput constraints for each CFDFC that
    // was marked to be optimized
    SmallVector<CFDFC *> cfdfcs;
    for (auto [cfdfc, optimize] : funcInfo.cfdfcs) {
      if (!optimize)
        continue;
      cfdfcs.push_back(cfdfc);
      addCFDFCVars(*cfdfc);
      addChannelThroughputConstraints(*cfdfc);
      addUnitThroughputConstraints(*cfdfc);
    }

    // Add the MILP objective and mark the MILP ready to be optimized
    addObjective(allChannels, cfdfcs);
    markReadyToOptimize();
    */
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
