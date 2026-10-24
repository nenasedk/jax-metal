
/* WARNING: Removing unreachable block (ram,0x00019648) */
/* WARNING: Removing unreachable block (ram,0x00019588) */
/* WARNING: Removing unreachable block (ram,0x00019528) */
/* WARNING: Removing unreachable block (ram,0x000195e8) */
/* WARNING: Removing unreachable block (ram,0x000196ac) */
/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* mlir::mps::populatePreLegalization(mlir::OpPassManager&, bool) */

void mlir::mps::populatePreLegalization(OpPassManager *param_1,bool param_2)

{
  long *plVar1;
  undefined8 uVar2;
  long *plStack_60;
  long *plStack_58;
  long *plStack_50;
  long *plStack_48;
  long *plStack_40;
  long *plStack_38;
  
  if (param_2) {
    mlir::mhlo::createStablehloLegalizeToHloPass();
    plStack_40 = plStack_38;
    plStack_38 = (long *)0x0;
    OpPassManager::addPass(param_1,&plStack_40);
    plVar1 = plStack_40;
    plStack_40 = (long *)0x0;
    if (plVar1 != (long *)0x0) {
      (**(code **)(*plVar1 + 8))();
    }
    plVar1 = plStack_38;
    plStack_38 = (long *)0x0;
    if (plVar1 != (long *)0x0) {
      (**(code **)(*plVar1 + 8))();
    }
  }
  createInlinerPass();
  OpPassManager::addPass(param_1,&plStack_48);
  plVar1 = plStack_48;
  plStack_48 = (long *)0x0;
  if (plVar1 != (long *)0x0) {
    (**(code **)(*plVar1 + 8))();
  }
  plVar1 = operator_new(0x158);
  plVar1[5] = 0;
  plVar1[4] = 0;
  plVar1[7] = 0;
  plVar1[6] = 0;
  plVar1[9] = 0;
  plVar1[8] = 0;
  plVar1[0xb] = 0;
  plVar1[10] = 0;
  plVar1[0xd] = 0;
  plVar1[0xc] = 0;
  plVar1[0xf] = 0;
  plVar1[0xe] = 0;
  plVar1[0x11] = 0;
  plVar1[0x10] = 0;
  plVar1[0x13] = 0;
  plVar1[0x12] = 0;
  plVar1[0x15] = 0;
  plVar1[0x14] = 0;
  plVar1[0x17] = 0;
  plVar1[0x16] = 0;
  plVar1[0x19] = 0;
  plVar1[0x18] = 0;
  plVar1[0x1b] = 0;
  plVar1[0x1a] = 0;
  plVar1[0x1d] = 0;
  plVar1[0x1c] = 0;
  plVar1[0x1f] = 0;
  plVar1[0x1e] = 0;
  plVar1[0x21] = 0;
  plVar1[0x20] = 0;
  plVar1[0x23] = 0;
  plVar1[0x22] = 0;
  plVar1[2] = (long)"func.func";
  plVar1[3] = 9;
  *(undefined1 *)(plVar1 + 4) = 1;
  plVar1[0x13] = 0;
  plVar1[0x12] = 0;
  plVar1[0x15] = 0;
  plVar1[0x14] = 0;
  plVar1[0x11] = 0;
  plVar1[0x10] = 0;
  plVar1[0x16] = 0;
  plVar1[0x17] = (long)(plVar1 + 0x19);
  plVar1[0x1d] = (long)(plVar1 + 0x1f);
  *(undefined4 *)((long)plVar1 + 0xc4) = 4;
  *(undefined4 *)((long)plVar1 + 0xf4) = 4;
  plVar1[0x23] = 0;
  plVar1[0x24] = 0;
  plVar1[0x25] = 0x1000000000;
  plVar1[0x27] = 0;
  plVar1[0x26] = 0;
  plVar1[0x29] = 0;
  plVar1[0x28] = 0;
  plVar1[0x2a] = 0;
  *plVar1 = (long)&PTR__DecomposeChloToHLOPass_05acdaf8;
  plVar1[1] = (long)&(anonymous_namespace)::DecomposeChloToHLOPass::resolveTypeID()::id;
  uVar2 = OpPassManager::nest(param_1,"func.func",9);
  plStack_38 = plVar1;
  OpPassManager::addPass(uVar2,&plStack_38);
  plVar1 = plStack_38;
  plStack_38 = (long *)0x0;
  if (plVar1 != (long *)0x0) {
    (**(code **)(*plVar1 + 8))();
  }
  mlir::mhlo::createLegalizeGeneralDotPass();
  uVar2 = OpPassManager::nest(param_1,"func.func",9);
  plStack_38 = plStack_50;
  OpPassManager::addPass(uVar2,&plStack_38);
  plVar1 = plStack_38;
  plStack_38 = (long *)0x0;
  if (plVar1 != (long *)0x0) {
    (**(code **)(*plVar1 + 8))();
  }
  mlir::mhlo::createHloCanonicalizeGatherPass();
  uVar2 = OpPassManager::nest(param_1,"func.func",9);
  plStack_38 = plStack_58;
  OpPassManager::addPass(uVar2,&plStack_38);
  plVar1 = plStack_38;
  plStack_38 = (long *)0x0;
  if (plVar1 != (long *)0x0) {
    (**(code **)(*plVar1 + 8))();
  }
  mlir::mhlo::createHloCanonicalizeScatterPass();
  uVar2 = OpPassManager::nest(param_1,"func.func",9);
  plStack_38 = (long *)0x0;
  OpPassManager::addPass(uVar2,&plStack_38);
  plVar1 = plStack_38;
  plStack_38 = (long *)0x0;
  if (plVar1 != (long *)0x0) {
    (**(code **)(*plVar1 + 8))();
  }
  mlir::mhlo::createFlattenTuplePass();
  uVar2 = OpPassManager::nest(param_1,"func.func",9);
  plStack_38 = (long *)0x0;
  OpPassManager::addPass(uVar2,&plStack_38);
  plVar1 = plStack_38;
  plStack_38 = (long *)0x0;
  if (plVar1 != (long *)0x0) {
    (**(code **)(*plVar1 + 8))();
  }
  createCanonicalizerPass();
  uVar2 = OpPassManager::nest(param_1,"func.func",9);
  plStack_38 = plStack_60;
  OpPassManager::addPass(uVar2,&plStack_38);
  plVar1 = plStack_38;
  plStack_38 = (long *)0x0;
  if (plVar1 != (long *)0x0) {
    (**(code **)(*plVar1 + 8))();
  }
  plVar1 = operator_new(0x158);
  plVar1[5] = 0;
  plVar1[4] = 0;
  plVar1[7] = 0;
  plVar1[6] = 0;
  plVar1[9] = 0;
  plVar1[8] = 0;
  plVar1[0xb] = 0;
  plVar1[10] = 0;
  plVar1[0xd] = 0;
  plVar1[0xc] = 0;
  plVar1[0xf] = 0;
  plVar1[0xe] = 0;
  plVar1[0x11] = 0;
  plVar1[0x10] = 0;
  plVar1[0x13] = 0;
  plVar1[0x12] = 0;
  plVar1[0x15] = 0;
  plVar1[0x14] = 0;
  plVar1[0x17] = 0;
  plVar1[0x16] = 0;
  plVar1[0x19] = 0;
  plVar1[0x18] = 0;
  plVar1[0x1b] = 0;
  plVar1[0x1a] = 0;
  plVar1[0x1d] = 0;
  plVar1[0x1c] = 0;
  plVar1[0x1f] = 0;
  plVar1[0x1e] = 0;
  plVar1[0x21] = 0;
  plVar1[0x20] = 0;
  plVar1[0x23] = 0;
  plVar1[0x22] = 0;
  plVar1[2] = (long)"func.func";
  plVar1[3] = 9;
  *(undefined1 *)(plVar1 + 4) = 1;
  plVar1[0x13] = 0;
  plVar1[0x12] = 0;
  plVar1[0x15] = 0;
  plVar1[0x14] = 0;
  plVar1[0x11] = 0;
  plVar1[0x10] = 0;
  plVar1[0x16] = 0;
  plVar1[0x17] = (long)(plVar1 + 0x19);
  plVar1[0x1d] = (long)(plVar1 + 0x1f);
  *(undefined4 *)((long)plVar1 + 0xc4) = 4;
  *(undefined4 *)((long)plVar1 + 0xf4) = 4;
  plVar1[0x23] = 0;
  plVar1[0x24] = 0;
  plVar1[0x25] = 0x1000000000;
  plVar1[0x27] = 0;
  plVar1[0x26] = 0;
  plVar1[0x29] = 0;
  plVar1[0x28] = 0;
  plVar1[0x2a] = 0;
  *plVar1 = (long)&PTR__LegalizeControlFlowPass_05acdcb8;
  plVar1[1] = (long)&(anonymous_namespace)::LegalizeControlFlowPass::resolveTypeID()::id;
  uVar2 = OpPassManager::nest(param_1,"func.func",9);
  plStack_38 = plVar1;
  OpPassManager::addPass(uVar2,&plStack_38);
  plVar1 = plStack_38;
  plStack_38 = (long *)0x0;
  if (plVar1 != (long *)0x0) {
    (**(code **)(*plVar1 + 8))();
  }
  return;
}


