
/* WARNING: Unknown calling convention -- yet parameter storage is locked */
/* pjrt::PJRT_Plugin_Attributes(PJRT_Plugin_Attributes_Args*) */

long * pjrt::PJRT_Plugin_Attributes(PJRT_Plugin_Attributes_Args *param_1)

{
  long *plVar1;
  long lStack_28;
  
  ActualStructSizeIsGreaterOrEqual
            (&lStack_28,"PJRT_Plugin_Attributes_Args",0x1b,0x18,*(undefined8 *)param_1);
  if (lStack_28 == 0) {
    plVar1 = (long *)0x0;
    *(undefined8 *)(param_1 + 0x18) = 0;
  }
  else {
    plVar1 = operator_new(8);
    *plVar1 = lStack_28;
  }
  return plVar1;
}


