//===- HandshakeEnums.h - Handshake enums declaration -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Handshake MLIR enums.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_ENUM_HANDSHAKE_HANDSHAKE_ENUMS_H
#define DYNAMATIC_ENUM_HANDSHAKE_HANDSHAKE_ENUMS_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

// Pull in all enum type definitions and utility function declarations.
#include "dynamatic/Dialect/Handshake/HandshakeEnums.h.inc"

#endif // DYNAMATIC_ENUM_HANDSHAKE_HANDSHAKE_ENUMS_H
