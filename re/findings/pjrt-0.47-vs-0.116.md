# structs in both headers: 110; unchanged 100, append-only 6, changed 4

## Changed in place (needs review)

### PJRT_Extension_Base
- [2] `PJRT_Extension_Base* next` -> `struct PJRT_Extension_Base* next`

### PJRT_Memory_Kind_Args
- [3] `const char* memory_kind` -> `const char* kind`
- [4] `size_t memory_kind_size` -> `size_t kind_size`

### PJRT_TopologyDescription_GetDeviceDescriptions_Args
- [2] `PJRT_TopologyDescription* topology` -> `const PJRT_TopologyDescription* topology`

### PJRT_TopologyDescription_PlatformName_Args
- [2] `PJRT_TopologyDescription* topology` -> `const PJRT_TopologyDescription* topology`

## Append-only (new trailing fields the plugin ignores)

- PJRT_Client_CreateViewOfDeviceBuffer_Args: `PJRT_Memory* memory`
- PJRT_Client_Create_Args: `PJRT_KeyValueTryGetCallback kv_try_get_callback`; `void* kv_try_get_user_arg`
- PJRT_Device_MemoryStats_Args: `int64_t peak_allocated_bytes`; `bool peak_allocated_bytes_is_set`
- PJRT_Executable_DeserializeAndLoad_Args: `const char* overridden_serialized_compile_options`; `size_t overridden_serialized_compile_options_size`; `PJRT_LoadOptions* load_options`
- PJRT_Executable_GetCompiledMemoryStats_Args: `int64_t peak_memory_in_bytes`; `int64_t total_size_in_bytes`; `int64_t total_allocation_bytes`; `int64_t indefinite_allocations`; `int64_t peak_unpadded_heap_bytes`
- PJRT_ExecuteOptions: `PJRT_ExecuteContext* context`; `const char* call_location`; `size_t num_tasks`; `int* task_ids`; `int64_t* incarnation_ids`; `PJRT_MultiSlice_Config* multi_slice_config`; `bool use_major_to_minor_data_layout_for_callbacks`; `PJRT_HloOutputCallbackInfo* hlo_output_callbacks`; `size_t num_hlo_output_callbacks`; `const PJRT_NamedValue* custom_options`; `size_t num_custom_options`

## PJRT_Api: plugin 97 entries, framework 138; prefix identical: True

Missing from the plugin: PJRT_Memory_Kind_Id, PJRT_ExecuteContext_Create, PJRT_ExecuteContext_Destroy, PJRT_Buffer_CopyRawToHost, PJRT_AsyncHostToDeviceTransferManager_Destroy, PJRT_AsyncHostToDeviceTransferManager_TransferData, PJRT_Client_CreateBuffersForAsyncHostToDevice, PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer, PJRT_AsyncHostToDeviceTransferManager_Device, PJRT_AsyncHostToDeviceTransferManager_BufferCount, PJRT_AsyncHostToDeviceTransferManager_BufferSize, PJRT_AsyncHostToDeviceTransferManager_SetBufferError, PJRT_AsyncHostToDeviceTransferManager_AddMetadata, PJRT_Client_DmaMap, PJRT_Client_DmaUnmap, PJRT_Client_CreateUninitializedBuffer, PJRT_Client_UpdateGlobalProcessInfo, PJRT_TopologyDescription_Deserialize, PJRT_Client_CreateAliasBuffer, PJRT_Client_FulfillAliasBuffer, PJRT_LoadedExecutable_GetDeviceAssignment, PJRT_Client_CreateErrorBuffer, PJRT_AsyncHostToDeviceTransferManager_TransferLiteral, PJRT_Buffer_CopyRawToHostFuture, PJRT_Device_PoisonExecution, PJRT_Device_CreateAsyncTrackingEvent, PJRT_AsyncTrackingEvent_Destroy, PJRT_Executable_GetCompileOptions, PJRT_Buffer_DonateWithControlDependency, PJRT_Event_Create, PJRT_Event_Set, PJRT_Device_GetAttributes, PJRT_Client_Load, PJRT_LoadedExecutable_AddressableDeviceLogicalIds, PJRT_Buffer_Bitcast, PJRT_Error_ForEachPayload, PJRT_TopologyDescription_Fingerprint, PJRT_Executable_ParameterMemoryKinds, PJRT_Device_ClearMemoryStats, PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace, PJRT_TopologyDescription_GetMemorySpaceKindIds
