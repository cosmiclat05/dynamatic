#!/bin/bash

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script arguments
DYNAMATIC_DIR="/home/oyasar/dynamatic"
SRC_DIR="/home/oyasar/dynamatic/integration-test/small/"
OUTPUT_DIR="/home/oyasar/dynamatic/integration-test/small/out/"
KERNEL_NAME="small"
#USE_SIMPLE_BUFFERS=$5
TARGET_CP=4.2
#POLYGEIST_PATH=$7
MAPBUF_BLIF_DIR=$8

source "$SRC_DIR"/utils.sh

POLYGEIST_CLANG_BIN="$DYNAMATIC_DIR/bin/cgeist"
CLANGXX_BIN="$DYNAMATIC_DIR/bin/clang++"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"
DYNAMATIC_PROFILER_BIN="$DYNAMATIC_DIR/bin/exp-frequency-profiler"
DYNAMATIC_EXPORT_DOT_BIN="$DYNAMATIC_DIR/bin/export-dot"
PYTHON_SCRIPT_MAPBUF="/home/oyasar/mapbuf_external/dist/Verilog/Verilog"

# Generated directories/files
COMP_DIR="$OUTPUT_DIR/comp"

F_HANDSHAKE_BUFFERED="$COMP_DIR/handshake_buffered.mlir"
F_HANDSHAKE_CUTLOOPBACKS="$COMP_DIR/handshake_cut_loopbacks.mlir"
F_HANDSHAKE_EXPORT="$COMP_DIR/handshake_export.mlir"
F_HW="$COMP_DIR/hw.mlir"
F_FREQUENCIES="$COMP_DIR/frequencies.csv"

# ============================================================================ #
# Helper funtions
# ============================================================================ #

# Exports Handshake-level IR to DOT using Dynamatic, then converts the DOT to
# a PNG using dot.
#   $1: input handshake-level IR filename
#   $1: output filename, without extension (will use .dot and .png)
export_dot() {
  local f_handshake="$1"
  local f_dot="$COMP_DIR/$2.dot"
  local f_png="$COMP_DIR/$2.png"

  # Export to DOT
  "$DYNAMATIC_EXPORT_DOT_BIN" "$f_handshake" "--edge-style=spline" \
    > "$f_dot"
  exit_on_fail "Failed to create $2 DOT" "Created $2 DOT"

  # Convert DOT graph to PNG
  dot -Tpng "$f_dot" > "$f_png"
  exit_on_fail "Failed to convert $2 DOT to PNG" "Converted $2 DOT to PNG"
  return 0
}

# ============================================================================ #
# Compilation flow
# ============================================================================ #

# Reset output directory
rm -rf "$COMP_DIR" && mkdir -p "$COMP_DIR"

# Smart buffer placement


"$PYTHON_SCRIPT_MAPBUF" "$KERNEL_NAME" "$OUTPUT_DIR"

echo_info "Running smart buffer placement with CP = $TARGET_CP"
cd "$COMP_DIR"
"$DYNAMATIC_OPT_BIN" "$SRC_DIR/handshake_buffered.mlir" \
  --handshake-set-buffering-properties="version=fpga20" \
  --handshake-place-buffers="algorithm=mapbuf frequencies=$SRC_DIR/frequencies.csv timing-models=$DYNAMATIC_DIR/data/components.json target-period=$TARGET_CP timeout=120 dump-logs blif-file=$OUTPUT_DIR/mapbuf/" \
  > "$F_HANDSHAKE_BUFFERED"
exit_on_fail "Failed to place smart buffers" "Placed smart buffers"

# handshake canonicalization
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_BUFFERED" \
  --handshake-canonicalize \
  --handshake-hoist-ext-instances \
  --handshake-reshape-channels \
  > "$F_HANDSHAKE_EXPORT"
exit_on_fail "Failed to canonicalize Handshake" "Canonicalized handshake"

# Export to DOT
export_dot "$F_HANDSHAKE_EXPORT" "$KERNEL_NAME"

# handshake level -> hw level
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" --lower-handshake-to-hw \
  > "$F_HW"
exit_on_fail "Failed to lower to HW" "Lowered to HW"

echo_info "Compilation succeeded"
