// PJRT shim that lets current JAX drive Apple's jax-metal 0.1.1 plugin.
//
// The plugin (pjrt_plugin_metal_14.dylib) implements PJRT C API 0.47 and reads
// StableHLO up to 0.19.4. This shim exposes the plugin's own function table with
// compatibility repairs (see shim/DESIGN.md):
//   1. The table lives in a zero-filled block far larger than the plugin's 808
//      bytes, so frameworks that read newer entry points without a version check
//      see null instead of whatever follows the plugin's table.
//   2. PJRT_Plugin_Attributes reports stablehlo_current_version 0.19.4, so the
//      framework downgrades programs to a version the plugin can deserialize.
//   3. Each device gets one synthetic "device" memory space. The plugin reports
//      none, but current frameworks route every host-to-device transfer through
//      the device's default memory space. Calls that take a memory are rewritten
//      to the equivalent device call before reaching the plugin.
//   4. Programs with inputs/outputs of types the plugin rejects at a program boundary
//      are compiled from a version without them (complex64 as float32[..., 2] pairs,
//      uint64 as int64, float8 and 4-bit integers as their bytes), with buffers relabeled
//      at execution (see below).
//   5. uint64, float8, float4 and 2-/4-bit integer buffers live in the plugin as a type
//      holding the same bytes (int64, uint8, int8); the shim reports their own type.
//   6. Empty input tensors use one-byte placeholders; their logical values are
//      reconstructed from metadata in the boundary program.
//   7. An optional memory-statistics entry returns UNIMPLEMENTED instead of being null.
//   8. One native client survives framework resets, avoiding a dangling compute stream.
// Every other entry is the plugin's own function pointer.

#include <dlfcn.h>
#include <libgen.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "pjrt_c_api.h"  // PJRT 0.116, the framework-side layout.

namespace {

// The package's __version__, passed in by setup.py and the Makefile.
#ifndef JAX_METAL_SHIM_VERSION
#define JAX_METAL_SHIM_VERSION "dev"
#endif
constexpr char kShimVersion[] = JAX_METAL_SHIM_VERSION;
constexpr char kPluginFileName[] = "pjrt_plugin_metal_14.dylib";

// The only plugin layout this shim knows how to copy.
constexpr int kPluginMajor = 0;
constexpr int kPluginMinor = 47;
constexpr size_t kPluginStructSize = 808;  // 40-byte header + 97 entries.

// Room for 512 entries; everything past the plugin's table stays null.
constexpr size_t kTableBytes = 512 * sizeof(void*);
static_assert(sizeof(PJRT_Api) <= kTableBytes, "PJRT_Api outgrew the padded table");
static_assert(kPluginStructSize <= sizeof(PJRT_Api), "plugin table larger than PJRT_Api");

alignas(PJRT_Api) unsigned char g_table[kTableBytes];
std::once_flag g_init_once;
const PJRT_Api* g_api = nullptr;
const PJRT_Api* g_plugin = nullptr;  // The plugin's own table, for forwarding.
bool g_trace = false;

// This optional entry is just beyond Apple's 808-byte table. Current jaxlib
// calls it without checking for null. Report unavailable statistics through
// PJRT's normal error contract; inventing zero byte counts would be misleading.
PJRT_Error g_memory_stats_unavailable{nullptr};
constexpr char kMemoryStatsUnavailable[] =
    "Apple's Metal plugin does not provide compiled memory statistics";

PJRT_Error* ShimCompiledMemoryStats(PJRT_Executable_GetCompiledMemoryStats_Args*) {
  return &g_memory_stats_unavailable;
}

void ShimErrorDestroy(PJRT_Error_Destroy_Args* args) {
  if (args->error != &g_memory_stats_unavailable) g_plugin->PJRT_Error_Destroy(args);
}

void ShimErrorMessage(PJRT_Error_Message_Args* args) {
  if (args->error != &g_memory_stats_unavailable) {
    g_plugin->PJRT_Error_Message(args);
    return;
  }
  args->message = kMemoryStatsUnavailable;
  args->message_size = sizeof(kMemoryStatsUnavailable) - 1;
}

PJRT_Error* ShimErrorGetCode(PJRT_Error_GetCode_Args* args) {
  if (args->error != &g_memory_stats_unavailable) return g_plugin->PJRT_Error_GetCode(args);
  args->code = PJRT_Error_Code_UNIMPLEMENTED;
  return nullptr;
}

// MetalStream::gComputeStream survives client destruction and points at a freed
// stream. Recreating a backend then dereferences it in memcpy_htod. Keep one
// native client across framework resets, rather than retaining a new client on
// every reset. Apple's create function ignores options and uses process-wide
// configuration; framework clients/devices are still recreated by jaxlib.
std::mutex g_client_mutex;
PJRT_Client* g_retained_client = nullptr;

void ReleaseRetainedClient() {
  if (!g_retained_client) return;
  PJRT_Client_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy.client = g_retained_client;
  if (PJRT_Error* error = g_plugin->PJRT_Client_Destroy(&destroy)) {
    PJRT_Error_Destroy_Args drop{};
    drop.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    drop.error = error;
    g_plugin->PJRT_Error_Destroy(&drop);
  }
  g_retained_client = nullptr;
}

PJRT_Error* ShimClientCreate(PJRT_Client_Create_Args* args) {
  std::lock_guard<std::mutex> lock(g_client_mutex);
  if (!g_retained_client) {
    if (PJRT_Error* error = g_plugin->PJRT_Client_Create(args)) return error;
    g_retained_client = args->client;
    std::atexit(ReleaseRetainedClient);
  }
  args->client = g_retained_client;
  return nullptr;
}

PJRT_Error* ShimClientDestroy(PJRT_Client_Destroy_Args* args) {
  std::lock_guard<std::mutex> lock(g_client_mutex);
  if (args->client == g_retained_client) return nullptr;
  return g_plugin->PJRT_Client_Destroy(args);
}

// StableHLO versions the plugin deserializes (from VhloToVersionPass in the
// plugin: current 0.19.4, minimum 0.9.0; see re/FINDINGS.md).
const int64_t kStablehloCurrent[] = {0, 19, 4};
const int64_t kStablehloMinimum[] = {0, 9, 0};

PJRT_NamedValue MakeInt64List(const char* name, const int64_t* values, size_t count) {
  PJRT_NamedValue value{};
  value.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  value.name = name;
  value.name_size = std::strlen(name);
  value.type = PJRT_NamedValue_kInt64List;
  value.int64_array_value = values;
  value.value_size = count;
  return value;
}

PJRT_NamedValue MakeString(const char* name, const char* text) {
  PJRT_NamedValue value{};
  value.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  value.name = name;
  value.name_size = std::strlen(name);
  value.type = PJRT_NamedValue_kString;
  value.string_value = text;
  value.value_size = std::strlen(text);
  return value;
}

const PJRT_NamedValue kAttributes[] = {
    MakeInt64List("stablehlo_current_version", kStablehloCurrent, 3),
    MakeInt64List("stablehlo_minimum_version", kStablehloMinimum, 3),
    MakeString("jax_metal_shim_version", kShimVersion),
};

PJRT_Error* ShimPluginAttributes(PJRT_Plugin_Attributes_Args* args) {
  args->attributes = kAttributes;
  args->num_attributes = sizeof(kAttributes) / sizeof(kAttributes[0]);
  if (g_trace) {
    std::fprintf(stderr, "[jax-metal-shim] PJRT_Plugin_Attributes -> stablehlo 0.19.4\n");
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Synthetic memory spaces: one "device" memory per device.

constexpr char kMemoryKind[] = "device";

struct ShimMemory {
  PJRT_Memory base;  // Must be first: frameworks read memory->vtable (null here).
  PJRT_Memory* self;
  PJRT_Device* device;
  int id;
  std::string debug_string;
  std::string to_string;
};

std::mutex g_memory_mutex;
std::unordered_map<PJRT_Device*, std::unique_ptr<ShimMemory>> g_device_memory;
std::unordered_map<const PJRT_Memory*, ShimMemory*> g_shim_memories;
std::unordered_map<PJRT_Client*, std::vector<PJRT_Memory*>> g_client_memories;

ShimMemory* MemoryForDevice(PJRT_Device* device) {
  std::lock_guard<std::mutex> lock(g_memory_mutex);
  std::unique_ptr<ShimMemory>& memory = g_device_memory[device];
  if (!memory) {
    memory = std::make_unique<ShimMemory>();
    memory->base.vtable = nullptr;
    memory->self = &memory->base;
    memory->device = device;
    memory->id = static_cast<int>(g_device_memory.size()) - 1;
    memory->debug_string = "METAL_DEVICE_MEMORY_" + std::to_string(memory->id);
    memory->to_string = "MemorySpace(kind=device, id=" + std::to_string(memory->id) + ")";
    g_shim_memories[&memory->base] = memory.get();
  }
  return memory.get();
}

// Returns null for memories the shim did not create.
ShimMemory* AsShimMemory(const PJRT_Memory* memory) {
  if (!memory) return nullptr;
  std::lock_guard<std::mutex> lock(g_memory_mutex);
  auto found = g_shim_memories.find(memory);
  return found == g_shim_memories.end() ? nullptr : found->second;
}

PJRT_Error* ShimClientAddressableMemories(PJRT_Client_AddressableMemories_Args* args) {
  PJRT_Client_AddressableDevices_Args devices{};
  devices.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  devices.client = args->client;
  if (PJRT_Error* error = g_plugin->PJRT_Client_AddressableDevices(&devices)) return error;
  std::vector<PJRT_Memory*> memories;
  for (size_t i = 0; i < devices.num_addressable_devices; ++i) {
    memories.push_back(MemoryForDevice(devices.addressable_devices[i])->self);
  }
  std::lock_guard<std::mutex> lock(g_memory_mutex);
  std::vector<PJRT_Memory*>& stored = g_client_memories[args->client];
  stored = std::move(memories);
  args->addressable_memories = stored.data();
  args->num_addressable_memories = stored.size();
  return nullptr;
}

PJRT_Error* ShimDeviceAddressableMemories(PJRT_Device_AddressableMemories_Args* args) {
  ShimMemory* memory = MemoryForDevice(args->device);
  args->memories = &memory->self;
  args->num_memories = 1;
  return nullptr;
}

PJRT_Error* ShimDeviceDefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
  args->memory = MemoryForDevice(args->device)->self;
  return nullptr;
}

PJRT_Error* ShimMemoryId(PJRT_Memory_Id_Args* args) {
  ShimMemory* memory = AsShimMemory(args->memory);
  if (!memory) return g_plugin->PJRT_Memory_Id(args);
  args->id = memory->id;
  return nullptr;
}

PJRT_Error* ShimMemoryKind(PJRT_Memory_Kind_Args* args) {
  if (!AsShimMemory(args->memory)) return g_plugin->PJRT_Memory_Kind(args);
  args->kind = kMemoryKind;
  args->kind_size = sizeof(kMemoryKind) - 1;
  return nullptr;
}

PJRT_Error* ShimMemoryDebugString(PJRT_Memory_DebugString_Args* args) {
  ShimMemory* memory = AsShimMemory(args->memory);
  if (!memory) return g_plugin->PJRT_Memory_DebugString(args);
  args->debug_string = memory->debug_string.c_str();
  args->debug_string_size = memory->debug_string.size();
  return nullptr;
}

PJRT_Error* ShimMemoryToString(PJRT_Memory_ToString_Args* args) {
  ShimMemory* memory = AsShimMemory(args->memory);
  if (!memory) return g_plugin->PJRT_Memory_ToString(args);
  args->to_string = memory->to_string.c_str();
  args->to_string_size = memory->to_string.size();
  return nullptr;
}

PJRT_Error* ShimMemoryAddressableByDevices(PJRT_Memory_AddressableByDevices_Args* args) {
  ShimMemory* memory = AsShimMemory(args->memory);
  if (!memory) return g_plugin->PJRT_Memory_AddressableByDevices(args);
  args->devices = &memory->device;
  args->num_devices = 1;
  return nullptr;
}

PJRT_Error* ShimBufferMemory(PJRT_Buffer_Memory_Args* args) {
  PJRT_Buffer_Device_Args device{};
  device.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
  device.buffer = args->buffer;
  if (PJRT_Error* error = g_plugin->PJRT_Buffer_Device(&device)) return error;
  args->memory = MemoryForDevice(device.device)->self;
  return nullptr;
}

// ---------------------------------------------------------------------------
// Retyped buffers: types the plugin cannot take at a program boundary but whose bytes a type
// it supports holds unchanged. Such buffers live in the plugin as the physical type; the
// shim remembers the logical type and reports it to the framework. Narrow types are one
// value per byte, the code in the low bits: frameworks transfer them unpacked.

PJRT_Buffer_Type PhysicalType(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type_U64:
      return PJRT_Buffer_Type_S64;
    case PJRT_Buffer_Type_F8E5M2:
    case PJRT_Buffer_Type_F8E4M3FN:
    case PJRT_Buffer_Type_F8E4M3B11FNUZ:
    case PJRT_Buffer_Type_F8E5M2FNUZ:
    case PJRT_Buffer_Type_F8E4M3FNUZ:
    case PJRT_Buffer_Type_F8E4M3:
    case PJRT_Buffer_Type_F8E3M4:
    case PJRT_Buffer_Type_F8E8M0FNU:
    case PJRT_Buffer_Type_F4E2M1FN:
    case PJRT_Buffer_Type_U4:
    case PJRT_Buffer_Type_U2:
      return PJRT_Buffer_Type_U8;
    case PJRT_Buffer_Type_S4:
    case PJRT_Buffer_Type_S2:
      return PJRT_Buffer_Type_S8;
    default:
      return type;
  }
}

std::mutex g_retyped_mutex;
std::unordered_map<PJRT_Buffer*, PJRT_Buffer_Type> g_retyped;

void Retype(PJRT_Buffer* buffer, PJRT_Buffer_Type logical) {
  if (!buffer || PhysicalType(logical) == logical) return;
  std::lock_guard<std::mutex> lock(g_retyped_mutex);
  g_retyped[buffer] = logical;
}

// The logical type of a retyped buffer, else `fallback`.
PJRT_Buffer_Type LogicalType(PJRT_Buffer* buffer, PJRT_Buffer_Type fallback) {
  std::lock_guard<std::mutex> lock(g_retyped_mutex);
  auto found = g_retyped.find(buffer);
  return found == g_retyped.end() ? fallback : found->second;
}

void Forget(PJRT_Buffer* buffer) {
  std::lock_guard<std::mutex> lock(g_retyped_mutex);
  g_retyped.erase(buffer);
}

PJRT_Error* ShimBufferElementType(PJRT_Buffer_ElementType_Args* args) {
  PJRT_Error* error = g_plugin->PJRT_Buffer_ElementType(args);
  if (!error) args->type = LogicalType(args->buffer, args->type);
  return error;
}

PJRT_Error* ShimBufferDestroy(PJRT_Buffer_Destroy_Args* args) {
  Forget(args->buffer);
  return g_plugin->PJRT_Buffer_Destroy(args);
}

PJRT_Error* ShimBufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
  // The plugin only understands devices and physical types; restore the caller's input
  // afterwards.
  PJRT_Buffer_Type logical = args->type;
  PJRT_Device* device = args->device;
  PJRT_Memory* requested = args->memory;
  if (ShimMemory* memory = AsShimMemory(args->memory)) {
    args->device = memory->device;
    args->memory = nullptr;
  }
  args->type = PhysicalType(logical);
  PJRT_Error* error = g_plugin->PJRT_Client_BufferFromHostBuffer(args);
  args->type = logical;
  args->device = device;
  args->memory = requested;
  if (!error) Retype(args->buffer, logical);
  return error;
}

PJRT_Error* ShimBufferCopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) {
  PJRT_Error* error = g_plugin->PJRT_Buffer_CopyToDevice(args);
  if (!error) Retype(args->dst_buffer, LogicalType(args->buffer, PJRT_Buffer_Type_INVALID));
  return error;
}

PJRT_Error* ShimBufferCopyToMemory(PJRT_Buffer_CopyToMemory_Args* args) {
  ShimMemory* memory = AsShimMemory(args->dst_memory);
  if (!memory) return g_plugin->PJRT_Buffer_CopyToMemory(args);
  PJRT_Buffer_CopyToDevice_Args copy{};
  copy.struct_size = PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE;
  copy.buffer = args->buffer;
  copy.dst_device = memory->device;
  PJRT_Error* error = ShimBufferCopyToDevice(&copy);
  args->dst_buffer = copy.dst_buffer;
  return error;
}

PJRT_Error* ShimCreateViewOfDeviceBuffer(PJRT_Client_CreateViewOfDeviceBuffer_Args* args) {
  // `memory` was appended after PJRT 0.47; the plugin ignores it.
  bool has_memory = args->struct_size >= PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
  ShimMemory* memory = has_memory ? AsShimMemory(args->memory) : nullptr;
  if (memory && !args->device) args->device = memory->device;
  return g_plugin->PJRT_Client_CreateViewOfDeviceBuffer(args);
}

// ---------------------------------------------------------------------------
// Program inputs and outputs of types the plugin rejects at a program boundary.
//
// The plugin rejects programs whose signature mentions complex, uint64, float8 or 4-bit
// integer types. For such a METAL program the shim's Python package lowers two versions: the
// original, which jaxlib compiles and takes its metadata from, and a boundary version where
// each complex64 value is a float32[..., 2] (real, imag) pair and each value of a retyped
// type (see "Retyped buffers") is its physical type, holding the same bytes. Right before
// compiling it hands the shim the serialized boundary program and the positions and kinds
// of the changed values (JaxMetalShim_SetBoundaryIo); the shim compiles the boundary program
// instead. At execution, complex64 inputs are passed as float32[..., 2] views of the same
// device memory and pair outputs are returned as complex64 views that own the pair buffers;
// retyped buffers already live as their physical type and such outputs are retyped. Buffers
// always match what jaxlib expects and nothing is copied.

// Kinds of boundary values; the Python package passes the same numbers.
enum BoundaryKind : int32_t {
  kComplexPairs = 1,  // complex64 <-> float32[..., 2]
  kEmptyInput = 2,    // empty tensor <-> unused scalar uint8 placeholder
  kRetyped = 0x100,   // | a PJRT_Buffer_Type T: T <-> PhysicalType(T)
};

PJRT_Buffer_Type RetypedType(int32_t kind) {
  return static_cast<PJRT_Buffer_Type>(kind & 0xFF);
}

struct BoundaryValue {
  int32_t position;
  int32_t kind;
};

struct BoundaryIo {
  std::vector<BoundaryValue> inputs;   // argument positions and kinds
  std::vector<BoundaryValue> outputs;  // output positions and kinds
  std::string program;                 // serialized boundary program (pending only)
  PJRT_Client* client = nullptr;
};

thread_local std::shared_ptr<BoundaryIo> t_pending_io;
std::mutex g_io_mutex;
std::unordered_map<PJRT_LoadedExecutable*, std::shared_ptr<BoundaryIo>> g_loaded_io;

std::shared_ptr<BoundaryIo> IoFor(PJRT_LoadedExecutable* executable) {
  std::lock_guard<std::mutex> lock(g_io_mutex);
  auto found = g_loaded_io.find(executable);
  return found == g_loaded_io.end() ? nullptr : found->second;
}

void Remember(PJRT_LoadedExecutable* executable, PJRT_Client* client) {
  auto io = std::make_shared<BoundaryIo>();
  io->inputs = t_pending_io->inputs;
  io->outputs = t_pending_io->outputs;
  io->client = client;
  std::lock_guard<std::mutex> lock(g_io_mutex);
  g_loaded_io[executable] = io;
}

void DropError(PJRT_Error* error) {
  if (!error) return;
  PJRT_Error_Destroy_Args drop{};
  drop.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  drop.error = error;
  g_plugin->PJRT_Error_Destroy(&drop);
}

void DestroyEvent(PJRT_Event* event) {
  PJRT_Event_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  destroy.event = event;
  DropError(g_plugin->PJRT_Event_Destroy(&destroy));
}

PJRT_Error* AwaitReady(PJRT_Buffer* buffer) {
  PJRT_Buffer_ReadyEvent_Args ready{};
  ready.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
  ready.buffer = buffer;
  if (PJRT_Error* error = g_plugin->PJRT_Buffer_ReadyEvent(&ready)) return error;
  PJRT_Event_Await_Args await_args{};
  await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  await_args.event = ready.event;
  PJRT_Error* error = g_plugin->PJRT_Event_Await(&await_args);
  DestroyEvent(ready.event);
  return error;
}

void DestroyBuffer(PJRT_Buffer* buffer) {
  PJRT_Buffer_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  destroy.buffer = buffer;
  DropError(g_plugin->PJRT_Buffer_Destroy(&destroy));
}

// A view of `buffer`'s memory as `type` with `dims`. The caller waits for `buffer` to be
// ready first: a view counts as ready as soon as it exists. If `owner` is set, deleting
// the view destroys it.
PJRT_Error* MakeView(PJRT_Client* client, PJRT_Buffer* buffer, PJRT_Buffer_Type type,
                     const std::vector<int64_t>& dims, PJRT_Buffer* owner, PJRT_Buffer** view) {
  PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args pointer{};
  pointer.struct_size = PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE;
  pointer.buffer = buffer;
  if (PJRT_Error* error = g_plugin->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&pointer)) {
    return error;
  }
  PJRT_Buffer_Device_Args device{};
  device.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
  device.buffer = buffer;
  if (PJRT_Error* error = g_plugin->PJRT_Buffer_Device(&device)) return error;
  PJRT_Client_CreateViewOfDeviceBuffer_Args create{};
  // The plugin implements PJRT 0.47; pass its struct size (no `memory` field).
  create.struct_size = offsetof(PJRT_Client_CreateViewOfDeviceBuffer_Args, memory);
  create.client = client;
  create.device_buffer_ptr = pointer.device_memory_ptr;
  create.dims = dims.data();
  create.num_dims = dims.size();
  create.element_type = type;
  create.device = device.device;
  // The plugin calls on_delete_callback unconditionally; a null one aborts the process
  // with std::bad_function_call.
  create.on_delete_callback = [](void*, void* original) {
    if (original) DestroyBuffer(static_cast<PJRT_Buffer*>(original));
  };
  create.on_delete_callback_arg = owner;
  if (PJRT_Error* error = g_plugin->PJRT_Client_CreateViewOfDeviceBuffer(&create)) return error;
  *view = create.buffer;
  return nullptr;
}

std::vector<int64_t> Dimensions(PJRT_Buffer* buffer, PJRT_Error** error) {
  PJRT_Buffer_Dimensions_Args dims{};
  dims.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  dims.buffer = buffer;
  *error = g_plugin->PJRT_Buffer_Dimensions(&dims);
  if (*error) return {};
  return std::vector<int64_t>(dims.dims, dims.dims + dims.num_dims);
}

PJRT_Error* ShimCompile(PJRT_Client_Compile_Args* args) {
  if (!t_pending_io) return g_plugin->PJRT_Client_Compile(args);
  // Compile the boundary program instead of the one jaxlib serialized.
  PJRT_Program program = *args->program;
  program.code = t_pending_io->program.data();
  program.code_size = t_pending_io->program.size();
  PJRT_Client_Compile_Args compile = *args;
  compile.program = &program;
  PJRT_Error* error = g_plugin->PJRT_Client_Compile(&compile);
  args->executable = compile.executable;
  if (!error) Remember(args->executable, args->client);
  return error;
}

// Executables loaded from JAX's persistent compilation cache skip PJRT_Client_Compile;
// they were compiled from the boundary program.
PJRT_Error* ShimDeserializeAndLoad(PJRT_Executable_DeserializeAndLoad_Args* args) {
  PJRT_Error* error = g_plugin->PJRT_Executable_DeserializeAndLoad(args);
  if (!error && t_pending_io) Remember(args->loaded_executable, args->client);
  return error;
}

PJRT_Error* ShimLoadedExecutableDestroy(PJRT_LoadedExecutable_Destroy_Args* args) {
  {
    std::lock_guard<std::mutex> lock(g_io_mutex);
    g_loaded_io.erase(args->executable);
  }
  return g_plugin->PJRT_LoadedExecutable_Destroy(args);
}

struct InputViews {
  std::vector<PJRT_Buffer*> views;
  void Release() {
    for (PJRT_Buffer* view : views) DestroyBuffer(view);
    views.clear();
  }
};

// Empty input values are reconstructed from shape/type metadata in the boundary
// program. MPSGraph still needs a nonempty argument buffer in their positions.
PJRT_Error* EmptyInputPlaceholder(PJRT_Client* client, PJRT_Buffer* original,
                                 PJRT_Buffer** result) {
  PJRT_Buffer_Device_Args device{};
  device.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
  device.buffer = original;
  if (PJRT_Error* error = g_plugin->PJRT_Buffer_Device(&device)) return error;
  static const uint8_t zero = 0;
  PJRT_Client_BufferFromHostBuffer_Args create{};
  create.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  create.client = client;
  create.data = &zero;
  create.type = PJRT_Buffer_Type_U8;
  create.device = device.device;
  create.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
  if (PJRT_Error* error = g_plugin->PJRT_Client_BufferFromHostBuffer(&create)) return error;
  if (create.done_with_host_buffer) DestroyEvent(create.done_with_host_buffer);
  *result = create.buffer;
  return nullptr;
}

PJRT_Error* ShimExecute(PJRT_LoadedExecutable_Execute_Args* args) {
  std::shared_ptr<BoundaryIo> io = IoFor(args->executable);
  if (!io) return g_plugin->PJRT_LoadedExecutable_Execute(args);

  // Swap complex64 inputs for float32[..., 2] views, and any retyped-type buffer the plugin
  // holds as that type (one the shim did not create) for a view as its physical type.
  auto views = new InputViews();
  std::vector<std::vector<PJRT_Buffer*>> lists(args->num_devices);
  std::vector<PJRT_Buffer* const*> list_pointers(args->num_devices);
  PJRT_Error* error = nullptr;
  for (size_t device = 0; device < args->num_devices && !error; ++device) {
    lists[device].assign(args->argument_lists[device], args->argument_lists[device] + args->num_args);
    for (const BoundaryValue& input : io->inputs) {
      if (static_cast<size_t>(input.position) >= args->num_args) continue;
      PJRT_Buffer* buffer = lists[device][input.position];
      if (input.kind == kEmptyInput) {
        PJRT_Buffer* placeholder = nullptr;
        if ((error = EmptyInputPlaceholder(io->client, buffer, &placeholder))) break;
        views->views.push_back(placeholder);
        lists[device][input.position] = placeholder;
        continue;
      }
      PJRT_Buffer_ElementType_Args type{};
      type.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
      type.buffer = buffer;
      if ((error = g_plugin->PJRT_Buffer_ElementType(&type))) break;  // the physical type
      PJRT_Buffer_Type view_type;
      if (input.kind == kComplexPairs && type.type == PJRT_Buffer_Type_C64) {
        view_type = PJRT_Buffer_Type_F32;
      } else if ((input.kind & kRetyped) && type.type == RetypedType(input.kind) &&
                 PhysicalType(type.type) != type.type) {
        view_type = PhysicalType(type.type);
      } else {
        continue;
      }
      std::vector<int64_t> dims = Dimensions(buffer, &error);
      if (error || (error = AwaitReady(buffer))) break;
      if (input.kind == kComplexPairs) dims.push_back(2);
      PJRT_Buffer* view = nullptr;
      if ((error = MakeView(io->client, buffer, view_type, dims, nullptr, &view))) break;
      views->views.push_back(view);
      lists[device][input.position] = view;
    }
    list_pointers[device] = lists[device].data();
  }
  if (error) {
    views->Release();
    delete views;
    return error;
  }

  // Run, keeping completion events so the views outlive the execution.
  std::vector<PJRT_Event*> own_events(args->num_devices, nullptr);
  bool caller_events = args->device_complete_events != nullptr;
  PJRT_LoadedExecutable_Execute_Args run = *args;
  run.argument_lists = list_pointers.data();
  if (!caller_events) run.device_complete_events = own_events.data();
  error = g_plugin->PJRT_LoadedExecutable_Execute(&run);
  if (error) {
    views->Release();
    delete views;
    return error;
  }
  PJRT_Event** events = caller_events ? args->device_complete_events : own_events.data();
  if (caller_events && !views->views.empty()) {
    PJRT_Event_OnReady_Args on_ready{};
    on_ready.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
    on_ready.event = events[0];
    on_ready.user_arg = views;
    on_ready.callback = [](PJRT_Error* event_error, void* user_arg) {
      auto* done = static_cast<InputViews*>(user_arg);
      done->Release();
      delete done;
      DropError(event_error);
    };
    if ((error = g_plugin->PJRT_Event_OnReady(&on_ready))) return error;
    views = nullptr;
  } else if (!caller_events) {
    for (PJRT_Event* event : own_events) {
      if (!event) continue;
      PJRT_Event_Await_Args await_args{};
      await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
      await_args.event = event;
      PJRT_Error* event_error = g_plugin->PJRT_Event_Await(&await_args);
      if (event_error && !error) {
        error = event_error;
      } else {
        DropError(event_error);
      }
      DestroyEvent(event);
    }
  }
  if (views) {
    views->Release();
    delete views;
  }
  if (error) return error;

  // Return float32[..., 2] outputs as complex64 views that own the pair buffers. Waiting
  // for the output first makes these programs synchronous: a view is ready on creation.
  // Outputs of retyped types are retyped in place.
  for (size_t device = 0; device < args->num_devices; ++device) {
    for (const BoundaryValue& output : io->outputs) {
      PJRT_Buffer*& buffer = args->output_lists[device][output.position];
      if (output.kind & kRetyped) {
        Retype(buffer, RetypedType(output.kind));
        continue;
      }
      std::vector<int64_t> dims = Dimensions(buffer, &error);
      if (error) return error;
      if (dims.empty() || dims.back() != 2) continue;
      if ((error = AwaitReady(buffer))) return error;
      dims.pop_back();
      PJRT_Buffer* view = nullptr;
      if ((error = MakeView(io->client, buffer, PJRT_Buffer_Type_C64, dims, buffer, &view))) {
        return error;
      }
      buffer = view;
    }
  }
  return nullptr;
}

// ---------------------------------------------------------------------------

std::string PluginPath() {
  if (const char* path = std::getenv("JAX_METAL_SHIM_PLUGIN_PATH"); path && *path) {
    return path;
  }
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&ShimPluginAttributes), &info) || !info.dli_fname) {
    return kPluginFileName;
  }
  std::string self = info.dli_fname;
  return std::string(dirname(self.data())) + "/" + kPluginFileName;
}

void Initialize() {
  const char* trace = std::getenv("JAX_METAL_SHIM_TRACE");
  g_trace = trace && *trace && std::strcmp(trace, "0") != 0;

  std::string path = PluginPath();
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    std::fprintf(stderr, "[jax-metal-shim] cannot load %s: %s\n", path.c_str(), dlerror());
    return;
  }
  using GetApiFn = const PJRT_Api* (*)();
  auto get_api = reinterpret_cast<GetApiFn>(dlsym(handle, "GetPjrtApi"));
  const PJRT_Api* plugin = get_api ? get_api() : nullptr;
  if (!plugin) {
    std::fprintf(stderr, "[jax-metal-shim] %s has no usable GetPjrtApi\n", path.c_str());
    return;
  }
  if (plugin->pjrt_api_version.major_version != kPluginMajor ||
      plugin->pjrt_api_version.minor_version != kPluginMinor ||
      plugin->struct_size != kPluginStructSize) {
    std::fprintf(stderr,
                 "[jax-metal-shim] %s reports PJRT %d.%d with struct_size %zu; "
                 "this shim supports only PJRT %d.%d (struct_size %zu, jax-metal 0.1.1)\n",
                 path.c_str(), plugin->pjrt_api_version.major_version,
                 plugin->pjrt_api_version.minor_version, plugin->struct_size, kPluginMajor,
                 kPluginMinor, kPluginStructSize);
    return;
  }

  std::memset(g_table, 0, sizeof(g_table));
  std::memcpy(g_table, plugin, kPluginStructSize);
  auto* api = reinterpret_cast<PJRT_Api*>(g_table);
  api->PJRT_Error_Destroy = ShimErrorDestroy;
  api->PJRT_Error_Message = ShimErrorMessage;
  api->PJRT_Error_GetCode = ShimErrorGetCode;
  api->PJRT_Executable_GetCompiledMemoryStats = ShimCompiledMemoryStats;
  api->PJRT_Client_Create = ShimClientCreate;
  api->PJRT_Client_Destroy = ShimClientDestroy;
  api->PJRT_Plugin_Attributes = ShimPluginAttributes;
  api->PJRT_Client_AddressableMemories = ShimClientAddressableMemories;
  api->PJRT_Device_AddressableMemories = ShimDeviceAddressableMemories;
  api->PJRT_Device_DefaultMemory = ShimDeviceDefaultMemory;
  api->PJRT_Memory_Id = ShimMemoryId;
  api->PJRT_Memory_Kind = ShimMemoryKind;
  api->PJRT_Memory_DebugString = ShimMemoryDebugString;
  api->PJRT_Memory_ToString = ShimMemoryToString;
  api->PJRT_Memory_AddressableByDevices = ShimMemoryAddressableByDevices;
  api->PJRT_Buffer_Memory = ShimBufferMemory;
  api->PJRT_Client_BufferFromHostBuffer = ShimBufferFromHostBuffer;
  api->PJRT_Buffer_CopyToDevice = ShimBufferCopyToDevice;
  api->PJRT_Buffer_CopyToMemory = ShimBufferCopyToMemory;
  api->PJRT_Buffer_ElementType = ShimBufferElementType;
  api->PJRT_Buffer_Destroy = ShimBufferDestroy;
  api->PJRT_Client_CreateViewOfDeviceBuffer = ShimCreateViewOfDeviceBuffer;
  api->PJRT_Client_Compile = ShimCompile;
  api->PJRT_Executable_DeserializeAndLoad = ShimDeserializeAndLoad;
  api->PJRT_LoadedExecutable_Destroy = ShimLoadedExecutableDestroy;
  api->PJRT_LoadedExecutable_Execute = ShimExecute;
  g_plugin = plugin;
  g_api = api;

  if (g_trace) {
    std::fprintf(stderr,
                 "[jax-metal-shim] %s: loaded %s (PJRT %d.%d, %zu-byte table, padded to %zu)\n",
                 kShimVersion, path.c_str(), api->pjrt_api_version.major_version,
                 api->pjrt_api_version.minor_version, api->struct_size, kTableBytes);
  }
}

}  // namespace

// Called by the shim's Python package (ctypes) around compilation: the next executable
// compiled or loaded on this thread has boundary values of these kinds (BoundaryKind) at
// these input/output positions, and `program` is its serialized boundary version. Pass a
// null program to clear.
extern "C" __attribute__((visibility("default"))) void JaxMetalShim_SetBoundaryIo(
    const int32_t* inputs, const int32_t* input_kinds, size_t num_inputs,
    const int32_t* outputs, const int32_t* output_kinds, size_t num_outputs,
    const char* program, size_t program_size) {
  if (!program) {
    t_pending_io.reset();
    return;
  }
  auto io = std::make_shared<BoundaryIo>();
  for (size_t i = 0; i < num_inputs; ++i) io->inputs.push_back({inputs[i], input_kinds[i]});
  for (size_t i = 0; i < num_outputs; ++i) io->outputs.push_back({outputs[i], output_kinds[i]});
  io->program.assign(program, program_size);
  t_pending_io = io;
}

extern "C" __attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
  std::call_once(g_init_once, Initialize);
  return g_api;
}
