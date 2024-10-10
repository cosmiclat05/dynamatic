module {
  handshake.func @small(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["idx", "start"], resNames = ["out0", "end"]} {
    %0 = source {handshake.bb = 0 : ui32, handshake.name = "source0"}
    %1 = constant %0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1000 : i11} : <i11>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %3 = mux %5 [%7, %2] {handshake.bb = 0 : ui32, handshake.name = "mux1"} : <i1>, <i32>
    %4 = ori %arg0, %3 {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.bb = 0 : ui32, handshake.name = "ori0"} : <i32>
    %5 = constant %0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = true} : <i1>
    %trueResult, %falseResult = cond_br %5, %4 {handshake.bb = 0 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %6 = buffer %trueResult {handshake.bb = 0 : ui32, handshake.name = "buffer0", hw.parameters = {NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %7 = buffer %6 {handshake.bb = 0 : ui32, handshake.name = "buffer1", hw.parameters = {NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {R: 1}>}} : <i32>
    end {bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.bb = 0 : ui32, handshake.name = "end0"} %falseResult, %arg1 : <i32>, <>
  }
}

