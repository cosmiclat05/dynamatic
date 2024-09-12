//===- MAPBUFBuffers.h - MAPBUF buffer placement ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

//
// This mainly declares the `MAPBUFPlacement` class, which inherits the abstract
// `BufferPlacementMILP` class to setup and solve a real MILP from which
// buffering decisions can be made. Every public member declared in this file is
// under the `dynamatic::buffer::mapbuf` namespace, as to not create name
// conflicts for common structs with other implementors of
// `BufferPlacementMILP`.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MAPBUFBUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MAPBUFBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "experimental/Support/CutEnumeration.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace dynamatic {
namespace buffer {
namespace mapbuf {

/// Holds the state and logic for MAPBUF smart buffer placement. To buffer a
/// dataflow circuit, this MILP-based algorithm creates:
/// 1. custom channel constraints derived from channel-specific buffering
///    properties
/// 2. path constraints for all non-memory channels and units
/// 3. elasticity constraints for all non-memory channels and units
/// 4. throughput constraints for all channels and units parts of CFDFCs that
///    were extracted from the function
/// 5. a maximixation objective, that rewards high CFDFC throughputs and
///    penalizes the placement of many large buffers in the circuit
class MAPBUFBuffers : public BufferPlacementMILP {
public:
  /// Setups the entire MILP that buffers the input dataflow circuit for the
  /// target clock period, after which (absent errors) it is ready for
  /// optimization. The `legacyPlacemnt` controls the interpretation of the
  /// MILP's results (non-legacy placement should yield faster circuits in
  /// general). If a channel's buffering properties are provably unsatisfiable,
  /// the MILP will not be marked ready for optimization, ensuring that further
  /// calls to `optimize` fail.
  MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
                double targetPeriod);

  /// Achieves the same as the other constructor but additionally logs placement
  /// decisions and achieved throughputs using the provided logger, and dumps
  /// the MILP model and solution at the provided name next to the log file.
  MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
                double targetPeriod, Logger &logger,
                StringRef milpName = "placement");

protected:
  /// Interprets the MILP solution to derive buffer placement decisions. Since
  /// the MILP cannot encode the placement of both opaque and transparent slots
  /// on a single channel, some "interpretation" of the results is necessary to
  /// derive "mixed" placements where some buffer slots are opaque and some are
  /// transparent. This interpretation is partically controlled by the
  /// `legacyPlacement` flag, and always respects the channel-specific buffering
  /// constraints.
  void extractResult(BufferPlacement &placement) override;

private:
  /// Adds channel-specific buffering constraints that were parsed from IR
  /// annotations to the Gurobi model.
  void addMapbufConstraints(Value channel);

  void addCustomChannelConstraints(Value channel);

  /// Setups the entire MILP, creating all variables, constraints, and setting
  /// the system's objective. Called by the constructor in the absence of prior
  /// failures, after which the MILP is ready to be optimized.
  void setup();
};

} // namespace mapbuf
} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MAPBUFBUFFERS_H