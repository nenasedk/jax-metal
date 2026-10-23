
/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* mlir::vhlo::(anonymous namespace)::VhloToVersionPass::runOnOperation() */

void mlir::vhlo::(anonymous_namespace)::VhloToVersionPass::runOnOperation(void)

{
  ulong uVar1;
  char cVar2;
  long in_x0;
  undefined ***pppuVar3;
  long lVar4;
  ulong uVar5;
  undefined8 uStack_260;
  undefined8 uStack_258;
  undefined8 uStack_250;
  undefined8 uStack_248;
  undefined8 uStack_240;
  Version VStack_230;
  int7 iStack_22f;
  undefined1 uStack_228;
  undefined7 uStack_227;
  undefined1 uStack_220;
  int7 iStack_21f;
  undefined1 uStack_210;
  int7 iStack_20f;
  undefined1 uStack_208;
  undefined7 uStack_207;
  undefined1 uStack_200;
  int7 iStack_1ff;
  char cStack_1f8;
  long lStack_1f0;
  ulong uStack_1e8;
  undefined2 uStack_1d0;
  undefined **ppuStack_1c8;
  Version *pVStack_1c0;
  undefined ***pppuStack_1b0;
  undefined **ppuStack_1a8;
  undefined8 uStack_1a0;
  undefined8 uStack_198;
  undefined4 uStack_190;
  undefined8 *puStack_188;
  undefined8 uStack_180;
  undefined8 uStack_178;
  undefined8 uStack_170;
  undefined4 uStack_168;
  undefined8 uStack_160;
  undefined8 uStack_158;
  undefined8 uStack_150;
  undefined8 uStack_148;
  undefined8 uStack_140;
  undefined8 uStack_138;
  undefined8 uStack_118;
  undefined8 uStack_110;
  char **ppcStack_108;
  ulong uStack_100;
  char *pcStack_f8;
  undefined8 uStack_f0;
  long lStack_48;
  
  lStack_48 = *(long *)PTR____stack_chk_guard_05acc2a8;
  uStack_110 = Attribute::getContext();
  uStack_1a0 = 0;
  uStack_198 = 0;
  ppuStack_1a8 = &PTR__ConversionTarget_05acda00;
  uStack_190 = 0;
  puStack_188 = &uStack_178;
  uStack_160 = 0;
  uStack_158 = 0;
  uStack_180 = 0;
  uStack_178 = 0;
  uStack_170 = 0;
  uStack_168 = 0;
  uStack_150 = 0x1000000000;
  uStack_148 = 0;
  uStack_140 = 0;
  uStack_138 = 0x2800000000;
  uStack_118 = 0;
  uVar1 = *(ulong *)(in_x0 + 0x1e0);
  lVar4 = *(long *)(in_x0 + 0x1d8);
  if (-1 < (char)*(byte *)(in_x0 + 0x1ef)) {
    uVar1 = (ulong)*(byte *)(in_x0 + 0x1ef);
    lVar4 = in_x0 + 0x1d8;
  }
  uVar5 = *(ulong *)(in_x0 + 0x28);
  Version::fromString(&uStack_210,lVar4,uVar1);
  uVar5 = uVar5 & 0xfffffffffffffff8;
  if (cStack_1f8 == '\0') {
    if (uVar1 == 0) {
      mlir::emitError(&ppcStack_108,*(undefined8 *)(uVar5 + 0x18));
      if (ppcStack_108 != (char **)0x0) {
        Diagnostic::operator<<((Diagnostic *)&uStack_100,"No target version specified.\n");
        if ((ppcStack_108 != (char **)0x0) &&
           (Diagnostic::operator<<
                      ((Diagnostic *)&uStack_100,"Target version must be of the form `#.#.#`."),
           ppcStack_108 != (char **)0x0)) {
          InFlightDiagnostic::report();
        }
      }
    }
    else {
      mlir::emitError(&ppcStack_108,*(undefined8 *)(uVar5 + 0x18));
      if (ppcStack_108 != (char **)0x0) {
        Diagnostic::operator<<((Diagnostic *)&uStack_100,"Invalid target version argument \'");
        if (ppcStack_108 != (char **)0x0) {
          uStack_1d0 = 0x105;
          lStack_1f0 = lVar4;
          uStack_1e8 = uVar1;
          Diagnostic::operator<<((Diagnostic *)&uStack_100,(Twine *)&lStack_1f0);
          if (((ppcStack_108 != (char **)0x0) &&
              (Diagnostic::operator<<((Diagnostic *)&uStack_100,"\'\n"),
              ppcStack_108 != (char **)0x0)) &&
             (Diagnostic::operator<<
                        ((Diagnostic *)&uStack_100,"Target version must be of the form `#.#.#`."),
             ppcStack_108 != (char **)0x0)) {
            InFlightDiagnostic::report();
          }
        }
      }
    }
LAB_03c7eb40:
    std::__optional_destruct_base<mlir::Diagnostic,false>::~__optional_destruct_base_abi_ne180100_
              ((__optional_destruct_base<mlir::Diagnostic,false> *)&uStack_100);
  }
  else {
    uStack_1e8 = CONCAT71(uStack_207,uStack_208);
    lStack_1f0 = CONCAT71(iStack_20f,uStack_210);
    if (iStack_20f < 0) {
LAB_03c7ea5c:
      mlir::emitError(&ppcStack_108,*(undefined8 *)(uVar5 + 0x18));
      if (ppcStack_108 != (char **)0x0) {
        Diagnostic::operator<<((Diagnostic *)&uStack_100,"target version ");
        if ((ppcStack_108 != (char **)0x0) &&
           (vhlo::operator<<((Diagnostic *)&uStack_100,(Version *)&lStack_1f0),
           ppcStack_108 != (char **)0x0)) {
          Diagnostic::operator<<((Diagnostic *)&uStack_100," is less than minimum supported ");
          uStack_228 = 9;
          uStack_227 = 0;
          VStack_230 = 0;
          iStack_22f = 0;
          uStack_220 = 0;
          iStack_21f = 0;
          if ((ppcStack_108 != (char **)0x0) &&
             (vhlo::operator<<((Diagnostic *)&uStack_100,&VStack_230), ppcStack_108 != (char **)0x0)
             ) {
            InFlightDiagnostic::report();
          }
        }
      }
      goto LAB_03c7eb40;
    }
    if (lStack_1f0 != 0) {
LAB_03c7e91c:
      mlir::emitError(&ppcStack_108,*(undefined8 *)(uVar5 + 0x18));
      if (ppcStack_108 != (char **)0x0) {
        Diagnostic::operator<<((Diagnostic *)&uStack_100,"target version ");
        if ((ppcStack_108 != (char **)0x0) &&
           (vhlo::operator<<((Diagnostic *)&uStack_100,(Version *)&lStack_1f0),
           ppcStack_108 != (char **)0x0)) {
          Diagnostic::operator<<((Diagnostic *)&uStack_100," is greater than current version ");
          uStack_228 = 0x13;
          uStack_227 = 0;
          VStack_230 = 0;
          iStack_22f = 0;
          uStack_220 = 4;
          iStack_21f = 0;
          if ((ppcStack_108 != (char **)0x0) &&
             (vhlo::operator<<((Diagnostic *)&uStack_100,&VStack_230), ppcStack_108 != (char **)0x0)
             ) {
            InFlightDiagnostic::report();
          }
        }
      }
      goto LAB_03c7eb40;
    }
    if (((long)uStack_1e8 < 9) || ((uStack_1e8 == 9 && (iStack_1ff < 0)))) goto LAB_03c7ea5c;
    if ((0x13 < uStack_1e8) || ((uStack_1e8 == 0x13 && (4 < CONCAT71(iStack_1ff,uStack_200)))))
    goto LAB_03c7e91c;
    uStack_227 = uStack_207;
    iStack_22f = iStack_20f;
    uStack_228 = uStack_208;
    uStack_220 = uStack_200;
    iStack_21f = iStack_1ff;
    VStack_230 = uStack_210;
    pVStack_1c0 = &VStack_230;
    ppuStack_1c8 = &PTR____func_05c7e658;
    pppuStack_1b0 = &ppuStack_1c8;
    pcStack_f8 = "vhlo";
    uStack_f0 = 4;
    uStack_100 = 0x200000001;
    ppcStack_108 = &pcStack_f8;
    ConversionTarget::setDialectAction(&ppuStack_1a8,&pcStack_f8,1,1);
    ConversionTarget::setLegalityCallback
              (&ppuStack_1a8,ppcStack_108,uStack_100 & 0xffffffff,&ppuStack_1c8);
    if (ppcStack_108 != &pcStack_f8) {
      _free();
    }
    if (pppuStack_1b0 == &ppuStack_1c8) {
      lVar4 = 4;
      pppuVar3 = &ppuStack_1c8;
LAB_03c7ec58:
      (*(code *)(*pppuVar3)[lVar4])();
    }
    else if (pppuStack_1b0 != (undefined ***)0x0) {
      lVar4 = 5;
      pppuVar3 = pppuStack_1b0;
      goto LAB_03c7ec58;
    }
    uStack_258 = 0;
    uStack_260 = 0;
    uStack_248 = 0;
    uStack_250 = 0;
    uStack_240 = 0;
    cVar2 = applyPartialConversion
                      (*(ulong *)(in_x0 + 0x28) & 0xfffffffffffffff8,&ppuStack_1a8,in_x0 + 0x468,
                       &uStack_260);
    if (cVar2 != '\0') goto LAB_03c7eb58;
  }
  *(ulong *)(in_x0 + 0x28) = *(ulong *)(in_x0 + 0x28) | 4;
LAB_03c7eb58:
  ConversionTarget::~ConversionTarget((ConversionTarget *)&ppuStack_1a8);
  if (*(long *)PTR____stack_chk_guard_05acc2a8 == lStack_48) {
    return;
  }
                    /* WARNING: Subroutine does not return */
  ___stack_chk_fail();
}


