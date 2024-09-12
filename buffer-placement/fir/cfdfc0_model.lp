\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
 [ 2 sArc_1_1 * varMaxExecs + 2 sArc_0_1 * varMaxExecs
   + 2 sArc_1_2 * varMaxExecs ] / 2 
Subject To
 arch_1_1: varMaxExecs <= 999
 arch_0_1: 998 sArc_0_1 + varMaxExecs <= 999
 arch_1_2: 998 sArc_1_2 + varMaxExecs <= 999
 oneBackedge: sArc_1_1 = 1
 in0: - sBB_0 = 0
 out0: - sBB_0 + sArc_0_1 = 0
 in1: - sBB_1 + sArc_1_1 + sArc_0_1 = 0
 out1: - sBB_1 + sArc_1_1 + sArc_1_2 = 0
 in2: - sBB_2 + sArc_1_2 = 0
 out2: - sBB_2 = 0
Bounds
 varMaxExecs <= 999
Binaries
 sBB_1 sBB_0 sBB_2 sArc_1_1 sArc_0_1 sArc_1_2
Generals
 varMaxExecs
End
