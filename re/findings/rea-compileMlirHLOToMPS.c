
/* WARNING: Removing unreachable block (ram,0x0001a2d0) */
/* WARNING: Removing unreachable block (ram,0x0001a12c) */

void _compileMlirHLOToMPS(undefined8 *param_1,undefined8 param_2,undefined8 param_3,int param_4)

{
  undefined ***pppuVar1;
  char cVar2;
  int iVar3;
  MLIRContext *pMVar4;
  undefined8 uVar5;
  long lVar6;
  long *plVar7;
  ulong uVar8;
  undefined **ppuStack_300;
  undefined8 uStack_2f8;
  undefined4 uStack_2f0;
  BytecodeWriterConfig aBStack_2e8 [8];
  long *plStack_2e0;
  undefined8 uStack_2d8;
  undefined8 uStack_2d0;
  undefined ***pppuStack_2c8;
  ulong uStack_2c0;
  undefined **ppuStack_2b8;
  undefined4 uStack_2b0;
  undefined8 uStack_2a8;
  undefined8 uStack_2a0;
  undefined8 uStack_298;
  undefined1 uStack_290;
  undefined8 uStack_288;
  undefined4 uStack_280;
  undefined8 uStack_278;
  long *plStack_1a0;
  char cStack_191;
  PassManager aPStack_190 [64];
  BaseScopedDiagnosticHandler aBStack_150 [248];
  long lStack_58;
  
  lStack_58 = *(long *)PTR____stack_chk_guard_05acc2a8;
  pMVar4 = (MLIRContext *)mlir::Attribute::getContext();
  mlir::BaseScopedDiagnosticHandler::BaseScopedDiagnosticHandler(aBStack_150,pMVar4,false,false);
  uVar5 = mlir::Attribute::getContext();
  mlir::PassManager::PassManager(aPStack_190,uVar5,"any",3,1);
  cStack_191 = '\0';
  ppuStack_2b8 = (undefined **)&cStack_191;
  mlir::detail::walk<mlir::ForwardIterator>
            (param_2,llvm::function_ref<mlir::WalkResult(mlir::Operation*)>::
                     callback_fn<compileMlirHLOToMPS::__0>,&ppuStack_2b8,1);
  if (cStack_191 != '\0') {
    uVar5 = mlir::Attribute::getContext();
    mlir::PassManager::PassManager((PassManager *)&ppuStack_2b8,uVar5,"any",3,1);
    mlir::mhlo::createStablehloLegalizeToHloPass();
    plStack_1a0 = plStack_2e0;
    plStack_2e0 = (long *)0x0;
    mlir::OpPassManager::addPass(&ppuStack_2b8,&plStack_1a0);
    plVar7 = plStack_1a0;
    plStack_1a0 = (long *)0x0;
    if (plVar7 != (long *)0x0) {
      (**(code **)(*plVar7 + 8))();
    }
    plVar7 = plStack_2e0;
    plStack_2e0 = (long *)0x0;
    if (plVar7 != (long *)0x0) {
      (**(code **)(*plVar7 + 8))();
    }
    cVar2 = mlir::PassManager::run((Operation *)&ppuStack_2b8);
    if (cVar2 == '\0') {
      tsl::errors::Internal<char_const*>("Unable to convert StableHLO to MHLO");
      mlir::PassManager::~PassManager((PassManager *)&ppuStack_2b8);
      goto LAB_0001a228;
    }
    mlir::PassManager::~PassManager((PassManager *)&ppuStack_2b8);
  }
  mlir::mps::populatePassManager((OpPassManager *)aPStack_190,false);
  lVar6 = _getenv("JAX_METAL_DUMP");
  cVar2 = mlir::PassManager::run((Operation *)aPStack_190);
  if (cVar2 == '\0') {
    if (((compileMlirHLOToMPS::$_1::operator()(int,char_const*)::vmodule_activated & 1) == 0) &&
       (iVar3 = ___cxa_guard_acquire
                          (&compileMlirHLOToMPS::$_1::operator()(int,char_const*)::vmodule_activated
                          ), iVar3 != 0)) {
      compileMlirHLOToMPS::$_1::operator()(int,char_const*)::vmodule_activated =
           tsl::internal::LogMessage::VmoduleActivated
                     ("pjrt_plugin/src/se/transforms/hlo_to_mps.cc",1);
      ___cxa_guard_release
                (&compileMlirHLOToMPS::$_1::operator()(int,char_const*)::vmodule_activated);
    }
    if (compileMlirHLOToMPS::$_1::operator()(int,char_const*)::vmodule_activated != '\0') {
      tsl::internal::LogMessage::LogMessage
                ((LogMessage *)&ppuStack_2b8,"pjrt_plugin/src/se/transforms/hlo_to_mps.cc",0x185,0);
      std::__put_character_sequence_abi_ne180100_<char,std::char_traits<char>>
                ((ostream *)&ppuStack_2b8,"MHLO->MPS lowering passes failed.",0x21);
      tsl::internal::LogMessage::~LogMessage((LogMessage *)&ppuStack_2b8);
    }
    if (lVar6 != 0) {
      mlir::Operation::dump();
    }
    mlir::BaseScopedDiagnosticHandler::ConsumeStatus();
    *param_1 = ppuStack_2b8;
    if ((((ulong)ppuStack_2b8 & 1) != 0) &&
       (*(int *)((long)ppuStack_2b8 + -1) = *(int *)((long)ppuStack_2b8 + -1) + 1,
       ((ulong)ppuStack_2b8 & 1) != 0)) {
      absl::lts_20230802::Status::UnrefNonInlined((ulong)ppuStack_2b8);
    }
  }
  else {
    if (((compileMlirHLOToMPS::$_2::operator()(int,char_const*)::vmodule_activated & 1) == 0) &&
       (iVar3 = ___cxa_guard_acquire
                          (&compileMlirHLOToMPS::$_2::operator()(int,char_const*)::vmodule_activated
                          ), iVar3 != 0)) {
      compileMlirHLOToMPS::$_2::operator()(int,char_const*)::vmodule_activated =
           tsl::internal::LogMessage::VmoduleActivated
                     ("pjrt_plugin/src/se/transforms/hlo_to_mps.cc",5);
      ___cxa_guard_release
                (&compileMlirHLOToMPS::$_2::operator()(int,char_const*)::vmodule_activated);
    }
    if ((lVar6 != 0) &&
       (compileMlirHLOToMPS::$_2::operator()(int,char_const*)::vmodule_activated != '\0')) {
      if (((compileMlirHLOToMPS::$_3::operator()(int,char_const*)::vmodule_activated & 1) == 0) &&
         (iVar3 = ___cxa_guard_acquire
                            (&compileMlirHLOToMPS::$_3::operator()(int,char_const*)::
                              vmodule_activated), iVar3 != 0)) {
        compileMlirHLOToMPS::$_3::operator()(int,char_const*)::vmodule_activated =
             tsl::internal::LogMessage::VmoduleActivated
                       ("pjrt_plugin/src/se/transforms/hlo_to_mps.cc",5);
        ___cxa_guard_release
                  (&compileMlirHLOToMPS::$_3::operator()(int,char_const*)::vmodule_activated);
      }
      if (compileMlirHLOToMPS::$_3::operator()(int,char_const*)::vmodule_activated != '\0') {
        tsl::internal::LogMessage::LogMessage
                  ((LogMessage *)&ppuStack_2b8,"pjrt_plugin/src/se/transforms/hlo_to_mps.cc",0x18b,0
                  );
        std::__put_character_sequence_abi_ne180100_<char,std::char_traits<char>>
                  ((ostream *)&ppuStack_2b8,"MPS module after conversion",0x1b);
        tsl::internal::LogMessage::~LogMessage((LogMessage *)&ppuStack_2b8);
      }
      mlir::Operation::dump();
    }
    if (param_4 != 0) {
      uStack_2b0 = 0;
      uStack_290 = 0;
      uStack_288 = 0;
      uStack_280 = 1;
      uStack_2a0 = 0;
      uStack_298 = 0;
      uStack_2a8 = 0;
      ppuStack_2b8 = &PTR__raw_string_ostream_05cb8740;
      uStack_278 = param_3;
      llvm::raw_ostream::SetBufferAndMode(&ppuStack_2b8,0,0,0);
      plStack_2e0 = (long *)0x0;
      uStack_2d8 = 0;
      uStack_2d0 = 0x1000000000;
      uStack_2c0 = 0;
      pppuStack_2c8 = &ppuStack_2b8;
      mlir::BytecodeWriterConfig::BytecodeWriterConfig(aBStack_2e8,&plStack_2e0,"MLIR19.0.0git",0xd)
      ;
      mlir::BytecodeWriterConfig::setDesiredBytecodeVersion((longlong)aBStack_2e8);
      ppuStack_300 = &PTR__MPSDialectVersion_05acd670;
      uStack_2f8 = 0x100000001;
      uStack_2f0 = 0;
      cVar2 = mlir::mps::serializeMPSModule(param_2,&ppuStack_300,&ppuStack_2b8,aBStack_2e8,0,0);
      if (cVar2 == '\0') {
        if (lVar6 != 0) {
          mlir::Operation::dump();
        }
        tsl::errors::Internal<char_const*>("Unable to serialize MPS module");
        mlir::BytecodeWriterConfig::~BytecodeWriterConfig(aBStack_2e8);
        pppuVar1 = pppuStack_2c8;
        if ((int)uStack_2c0 != 0) {
          lVar6 = (uStack_2c0 & 0xffffffff) << 5;
          do {
            plVar7 = *(long **)((long)pppuVar1 + lVar6 + -8);
            *(undefined8 *)((long)pppuVar1 + lVar6 + -8) = 0;
            if (plVar7 != (long *)0x0) {
              (**(code **)(*plVar7 + 8))();
            }
            lVar6 = lVar6 + -0x20;
          } while (lVar6 != 0);
        }
        if (pppuStack_2c8 != &ppuStack_2b8) {
          _free(pppuStack_2c8);
        }
        if (uStack_2d8._4_4_ != 0) {
          uVar8 = uStack_2d8 & 0xffffffff;
          if ((int)uStack_2d8 != 0) {
            lVar6 = 0;
            do {
              plVar7 = *(long **)((long)plStack_2e0 + lVar6);
              if (plVar7 != (long *)&DAT_fffffffffffffff8 && plVar7 != (long *)0x0) {
                llvm::deallocate_buffer(plVar7,*plVar7 + 0x11,8);
              }
              lVar6 = lVar6 + 8;
            } while (uVar8 * 8 - lVar6 != 0);
          }
        }
        _free(plStack_2e0);
        llvm::raw_ostream::~raw_ostream((raw_ostream *)&ppuStack_2b8);
        goto LAB_0001a228;
      }
      mlir::BytecodeWriterConfig::~BytecodeWriterConfig(aBStack_2e8);
      pppuVar1 = pppuStack_2c8;
      if ((int)uStack_2c0 != 0) {
        lVar6 = (uStack_2c0 & 0xffffffff) << 5;
        do {
          plVar7 = *(long **)((long)pppuVar1 + lVar6 + -8);
          *(undefined8 *)((long)pppuVar1 + lVar6 + -8) = 0;
          if (plVar7 != (long *)0x0) {
            (**(code **)(*plVar7 + 8))();
          }
          lVar6 = lVar6 + -0x20;
        } while (lVar6 != 0);
      }
      if (pppuStack_2c8 != &ppuStack_2b8) {
        _free(pppuStack_2c8);
      }
      if (uStack_2d8._4_4_ != 0) {
        uVar8 = uStack_2d8 & 0xffffffff;
        if ((int)uStack_2d8 != 0) {
          lVar6 = 0;
          do {
            plVar7 = *(long **)((long)plStack_2e0 + lVar6);
            if (plVar7 != (long *)&DAT_fffffffffffffff8 && plVar7 != (long *)0x0) {
              llvm::deallocate_buffer(plVar7,*plVar7 + 0x11,8);
            }
            lVar6 = lVar6 + 8;
          } while (uVar8 * 8 - lVar6 != 0);
        }
      }
      _free(plStack_2e0);
      llvm::raw_ostream::~raw_ostream((raw_ostream *)&ppuStack_2b8);
    }
    *param_1 = 0;
  }
LAB_0001a228:
  mlir::PassManager::~PassManager(aPStack_190);
  mlir::BaseScopedDiagnosticHandler::~BaseScopedDiagnosticHandler(aBStack_150);
  if (*(long *)PTR____stack_chk_guard_05acc2a8 != lStack_58) {
                    /* WARNING: Subroutine does not return */
    ___stack_chk_fail();
  }
  return;
}


