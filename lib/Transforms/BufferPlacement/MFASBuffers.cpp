//===- MFASBuffers.cpp - MFAS buffer placement -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements Minimum Feedback Arc Set algorithm for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/MFASBuffers.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "mlir/IR/Value.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::mfas;

MFASBuffers::MFASBuffers(GRBEnv &env, FuncInfo &funcInfo,
                         const TimingDatabase &timingDB, double targetPeriod)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod) {
  if (!unsatisfiable)
    setup();
}

MFASBuffers::MFASBuffers(GRBEnv &env, FuncInfo &funcInfo,
                         const TimingDatabase &timingDB, double targetPeriod,
                         Logger &logger, StringRef milpName)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                          milpName) {
  if (!unsatisfiable)
    setup();
}

void MFASBuffers::extractResult(BufferPlacement &placement) {
  for (auto &[channel, props] : channelProps) {
    ChannelVars &channelVars = vars.channelVars[channel];
    for (auto &[sig, signalVars] : channelVars.signalVars) {
      auto dataIt = channelVars.signalVars.find(SignalType::DATA);
      if (dataIt != channelVars.signalVars.end()) {
        GRBVar &dataBuf = dataIt->second.bufPresent;
        if (dataBuf.get(GRB_DoubleAttr_X) > 0) {
          if (props.maxTrans.value_or(1) >= 1) {
            props.minTrans = std::max(props.minTrans, 1U);
          }
          if (props.maxOpaque.value_or(1) >= 1) {
            props.minOpaque = std::max(props.minOpaque, 1U);
          }
        }
      }
    }

    PlacementResult result{props.minTrans, props.minOpaque};
    result.deductInternalBuffers(Channel(channel), timingDB);
    placement[channel] = result;
  }

  if (logger)
    logResults(placement);
}

void MFASBuffers::addObjective() {
  // Minimize the number of edges that needs to be removed
  GRBLinExpr obj = 0;

  for (auto &[channel, _] : channelProps) {
    ChannelVars &channelVars = vars.channelVars[channel];
    for (auto &[_, signalVars] : channelVars.signalVars) {
      auto dataIt = channelVars.signalVars.find(SignalType::DATA);
      if (dataIt != channelVars.signalVars.end()) {
        GRBVar &dataBuf = dataIt->second.bufPresent;
        // If there is a data buffer on the channel, the channel elastic
        // arrival time at the ouput must be greater than at the input
        obj += dataBuf;
      }
    }
  }

  model.setObjective(obj, GRB_MINIMIZE);
}

void MFASBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 1> signals;
  signals.push_back(SignalType::DATA);

  /// NOTE: (lucas-rami) For each buffering group this should be the timing
  /// model of the buffer that will be inserted by the MILP for this group. We
  /// don't have models for these buffers at the moment therefore we provide a
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

    // Add path and elasticity constraints over all channels in the function
    // that are not adjacent to a memory interface
    if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
        !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
      addChannelElasticityConstraints(channel, bufGroups);
    }
  }

  // Add path and elasticity constraints over all units in the function
  for (Operation &op : funcInfo.funcOp.getOps()) {
    addUnitElasticityConstraints(&op);
  }

  // Add the MILP objective and mark the MILP ready to be optimized
  addObjective();
  markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
