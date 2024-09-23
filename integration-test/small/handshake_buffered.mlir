module {
  handshake.func @small(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["idx", "start"], resNames = ["out0", "end"]} {
    %0 = source {handshake.bb = 0 : ui32, handshake.name = "source0"}
    %1 = constant %0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 1000 : i11} : <i11>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %10 = source {handshake.bb = 0 : ui32, handshake.name = "source1"}
    %11 = constant %10 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = true} : <i1>
    %6 = mux %11 [%5, %2] {handshake.bb = 0 : ui32, handshake.name = "mux1"} : <i1>, <i32>
    %3 = addi %arg0, %6 {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.bb = 0 : ui32, handshake.name = "addi0"} : <i32>
    %8 = source {handshake.bb = 0 : ui32, handshake.name = "source2"}
    %7 = constant %8 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = true} : <i1>
    %trueResult_0, %falseResult_0 = cond_br %7, %3 {handshake.bb = 0 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %4 = buffer %trueResult_0 {handshake.bb = 0 : ui32, handshake.name = "buffer0", hw.parameters = {NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %5 = buffer %4 {handshake.bb = 0 : ui32, handshake.name = "buffer1", hw.parameters = {NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {R: 1}>}} : <i32>
    end {bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.bb = 0 : ui32, handshake.name = "end0"} %falseResult_0, %arg1 : <i32>, <>
  }
}
