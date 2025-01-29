//===- MFASBuffers.h - Minimum Feedback Arc Set buffer placement ---*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Finds the Minimum Feedback Arc Set (MFAS) in a dataflow graph, and places
// buffer on this arc set to minimize the number of buffers.
//
// This mainly declares the `MFAS` class, which inherits the abstract
// `BufferPlacementMILP` class to setup and solve a real MILP from which
// buffering decisions can be made. Every public member declared in this file is
// under the `dynamatic::buffer::mfas` namespace, as to not create name
// conflicts for common structs with other implementors of
// `BufferPlacementMILP`.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MFASBUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MFASBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace dynamatic {
namespace buffer {
namespace mfas {

/// Holds the state and logic for FPGA'20 smart buffer placement. To buffer a
/// dataflow circuit, this MILP-based algorithm creates:
/// 1. custom channel constraints derived from channel-specific buffering
///    properties
/// 2. path constraints for all non-memory channels and units
/// 3. elasticity constraints for all non-memory channels and units
/// 4. throughput constraints for all channels and units parts of CFDFCs that
///    were extracted from the function
/// 5. a maximixation objective, that rewards high CFDFC throughputs and
///    penalizes the placement of many large buffers in the circuit

/// Holds the state and logic for MFAS buffer placement.
/// 1. Adds elasticity constraints
class MFASBuffers : public BufferPlacementMILP {
public:
  /// Setups the entire MILP that buffers the input dataflow circuit for the
  /// target clock period, after which (absent errors) it is ready for
  /// optimization.
  MFASBuffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
              double targetPeriod);

  /// Achieves the same as the other constructor but additionally logs placement
  /// decisions and achieved throughputs using the provided logger, and dumps
  /// the MILP model and solution at the provided name next to the log file.
  MFASBuffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
              double targetPeriod, Logger &logger,
              StringRef milpName = "placement");

protected:
  /// Interprets the MILP solution to derive buffer placement decisions. Places
  /// 1 opaque and 1 transparent buffer on the channels that are in minimum
  /// feedback arc set.
  void extractResult(BufferPlacement &placement) override;

  /// Adds the MILP model's objective to maximize. The objective is to minimize
  /// the number of edges that needs to be removed.
  void addObjective();

private:
  /// Setups the entire MILP, creating all variables, constraints, and setting
  /// the system's objective. Called by the constructor in the absence of prior
  /// failures, after which the MILP is ready to be optimized.
  void setup();
};

} // namespace mfas
} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MFASBUFFERS_H