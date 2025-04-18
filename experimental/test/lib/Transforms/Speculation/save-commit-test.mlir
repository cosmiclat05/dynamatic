// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt %s --handshake-speculation="json-path=%S/save-commit-test.json automatic=false"

// CHECK-LABEL:   handshake.func @placeSaveCommitsOnAllPaths(
// CHECK-SAME:                                               %[[VAL_0:.*]]: none, ...) attributes {argNames = ["start"], resNames = []} {
// CHECK:           %[[VAL_1:.*]]:2 = fork [2] %[[VAL_0]] {handshake.bb = 0 : ui32, handshake.name = "fork1"} : none
// CHECK:           %[[VAL_2:.*]] = constant %[[VAL_1]]#0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, i1
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = control_merge %[[VAL_5:.*]], %[[VAL_2]] {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : i1, i1
// CHECK:           %[[VAL_6:.*]] = spec_save_commit{{\[}}%[[VAL_7:.*]]] %[[VAL_3]] {handshake.bb = 1 : ui32, handshake.name = "spec_save_commit0"} : i1
// CHECK:           %[[VAL_8:.*]] = spec_save_commit{{\[}}%[[VAL_7]]] %[[VAL_9:.*]]#2 {handshake.bb = 1 : ui32, handshake.name = "spec_save_commit1"} : i1
// CHECK:           %[[VAL_5]], %[[VAL_10:.*]] = cond_br %[[VAL_8]], %[[VAL_6]] {handshake.bb = 1 : ui32, handshake.name = "cond_br0"} : i1
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_1]]#1 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = true} : <>, i1
// CHECK:           %[[VAL_12:.*]] = mux %[[VAL_4]] {{\[}}%[[VAL_13:.*]], %[[VAL_11]]] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : i1, i1
// CHECK:           %[[VAL_9]]:3 = fork [3] %[[VAL_12]] {handshake.bb = 1 : ui32, handshake.name = "fork0"} : i1
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]], %[[VAL_16:.*]], %[[VAL_17:.*]], %[[VAL_18:.*]], %[[VAL_19:.*]] = speculator{{\[}}%[[VAL_1]]#1] %[[VAL_9]]#1 {handshake.bb = 1 : ui32, handshake.name = "speculator0"} : i1
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]] = speculating_branch{{\[}}%[[VAL_14]]] %[[VAL_8]] {handshake.bb = 1 : ui32, handshake.name = "speculating_branch0"} : i1, i1
// CHECK:           %[[VAL_22:.*]], %[[VAL_23:.*]] = cond_br %[[VAL_20]], %[[VAL_18]] {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : i3
// CHECK:           %[[VAL_24:.*]], %[[VAL_25:.*]] = cond_br %[[VAL_19]], %[[VAL_22]] {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : i3
// CHECK:           %[[VAL_7]] = merge %[[VAL_17]], %[[VAL_24]] {handshake.bb = 1 : ui32, handshake.name = "merge0"} : i3
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = speculating_branch{{\[}}%[[VAL_14]]] %[[VAL_28:.*]] {handshake.bb = 1 : ui32, handshake.name = "speculating_branch1"} : i1, i1
// CHECK:           %[[VAL_29:.*]], %[[VAL_30:.*]] = cond_br %[[VAL_26]], %[[VAL_16]] {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : i1
// CHECK:           %[[VAL_31:.*]], %[[VAL_32:.*]] = speculating_branch{{\[}}%[[VAL_8]]] %[[VAL_8]] {handshake.bb = 1 : ui32, handshake.name = "speculating_branch2"} : i1, i1
// CHECK:           %[[VAL_33:.*]], %[[VAL_34:.*]] = cond_br %[[VAL_31]], %[[VAL_29]] {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : i1
// CHECK:           %[[VAL_28]] = spec_save_commit{{\[}}%[[VAL_7]]] %[[VAL_9]]#0 {handshake.bb = 1 : ui32, handshake.name = "spec_save_commit2"} : i1
// CHECK:           %[[VAL_13]], %[[VAL_35:.*]] = cond_br %[[VAL_28]], %[[VAL_14]] {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : i1
// CHECK:           %[[VAL_36:.*]] = spec_commit{{\[}}%[[VAL_34]]] %[[VAL_10]] {handshake.bb = 1 : ui32, handshake.name = "spec_commit0"} : i1
// CHECK:           end {handshake.bb = 1 : ui32, handshake.name = "end0"} %[[VAL_36]] : i1
// CHECK:         }
handshake.func @placeSaveCommitsOnAllPaths(%start: !handshake.control<>) -> !handshake.channel<i1> {
  %0:2 =  fork [2] %start  {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <>
  %4 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 0 : i1} : <>, <i1>
  %result, %index = control_merge %trueResult, %4 {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : <i1>, <i1>
  %trueResult, %falseResult = cond_br %3#2, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br0"} : <i1>, <i1>
  %1 = constant %0#1 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = 1 : i1} : <>, <i1>
  %2 = mux %index [%trueResult1, %1] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, <i1>
  %3:3 = fork [3] %2  {handshake.bb = 1 : ui32, handshake.name = "fork0"} : <i1>
  %trueResult1, %falseResult1 = cond_br %3#0, %3#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i1>
  end {handshake.bb = 1 : ui32, handshake.name =  "end0"} %falseResult  : <i1>
}
