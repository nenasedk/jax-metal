// A stand-in for Apple's plugin with just enough of PJRT 0.47 to test the shim's buffer
// retyping and boundary programs on any machine (shim_boundary_test.cc).
//
// Buffers are host memory. Compile remembers the program it was given; Execute records the
// element types and shapes of its arguments and returns three outputs: int64[3],
// float32[3, 2] and uint8[3].

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "pjrt_c_api.h"  // PJRT 0.47, the plugin-side layout.

struct PJRT_Client {};
struct PJRT_Device {};
struct PJRT_Event {};
struct PJRT_Error {};
struct PJRT_LoadedExecutable {
  std::string program;
};
struct PJRT_Buffer {
  PJRT_Buffer_Type type;
  std::vector<int64_t> dims;
  std::shared_ptr<std::vector<char>> data;
  void (*on_delete)(void*, void*) = nullptr;
  void* on_delete_arg = nullptr;
};

namespace {

PJRT_Device g_device;
PJRT_Event g_event;
int g_client_creates = 0, g_client_destroys = 0;
std::string g_last_program;
std::vector<PJRT_Buffer_Type> g_last_arg_types;
std::vector<std::vector<int64_t>> g_last_arg_dims;

PJRT_Error* ClientCreate(PJRT_Client_Create_Args* args) {
  ++g_client_creates;
  args->client = new PJRT_Client;
  return nullptr;
}

PJRT_Error* ClientDestroy(PJRT_Client_Destroy_Args* args) {
  if (args->client) ++g_client_destroys;
  delete args->client;
  return nullptr;
}

size_t ElementSize(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type_S8: case PJRT_Buffer_Type_U8: return 1;
    case PJRT_Buffer_Type_F32: case PJRT_Buffer_Type_S32: case PJRT_Buffer_Type_U32: return 4;
    default: return 8;
  }
}

PJRT_Buffer* NewBuffer(PJRT_Buffer_Type type, std::vector<int64_t> dims) {
  size_t count = 1;
  for (int64_t d : dims) count *= d;
  auto* buffer = new PJRT_Buffer{type, std::move(dims), nullptr};
  buffer->data = std::make_shared<std::vector<char>>(count * ElementSize(type));
  return buffer;
}

PJRT_Error* BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
  PJRT_Buffer* buffer = NewBuffer(args->type, {args->dims, args->dims + args->num_dims});
  if (!buffer->data->empty()) {
    std::memcpy(buffer->data->data(), args->data, buffer->data->size());
  }
  args->buffer = buffer;
  args->done_with_host_buffer = nullptr;
  return nullptr;
}

PJRT_Error* ElementType(PJRT_Buffer_ElementType_Args* args) {
  args->type = args->buffer->type;
  return nullptr;
}

PJRT_Error* Dimensions(PJRT_Buffer_Dimensions_Args* args) {
  args->dims = args->buffer->dims.data();
  args->num_dims = args->buffer->dims.size();
  return nullptr;
}

PJRT_Error* Destroy(PJRT_Buffer_Destroy_Args* args) {
  if (args->buffer->on_delete) args->buffer->on_delete(nullptr, args->buffer->on_delete_arg);
  delete args->buffer;
  return nullptr;
}

PJRT_Error* CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) {
  PJRT_Buffer* copy = NewBuffer(args->buffer->type, args->buffer->dims);
  *copy->data = *args->buffer->data;
  args->dst_buffer = copy;
  return nullptr;
}

PJRT_Error* DataPointer(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args) {
  args->device_memory_ptr = args->buffer->data.get();  // views share the storage object
  return nullptr;
}

PJRT_Error* BufferDevice(PJRT_Buffer_Device_Args* args) {
  args->device = &g_device;
  return nullptr;
}

PJRT_Error* CreateView(PJRT_Client_CreateViewOfDeviceBuffer_Args* args) {
  auto* storage = static_cast<std::vector<char>*>(args->device_buffer_ptr);
  auto* view = new PJRT_Buffer{args->element_type, {args->dims, args->dims + args->num_dims},
                               std::shared_ptr<std::vector<char>>(storage, [](auto*) {})};
  view->on_delete = args->on_delete_callback;
  view->on_delete_arg = args->on_delete_callback_arg;
  args->buffer = view;
  return nullptr;
}

PJRT_Error* ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
  args->event = &g_event;
  return nullptr;
}

PJRT_Error* EventAwait(PJRT_Event_Await_Args*) { return nullptr; }
PJRT_Error* EventDestroy(PJRT_Event_Destroy_Args*) { return nullptr; }
void ErrorDestroy(PJRT_Error_Destroy_Args*) {}

PJRT_Error* Compile(PJRT_Client_Compile_Args* args) {
  g_last_program.assign(args->program->code, args->program->code_size);
  args->executable = new PJRT_LoadedExecutable{g_last_program};
  return nullptr;
}

PJRT_Error* ExecutableDestroy(PJRT_LoadedExecutable_Destroy_Args* args) {
  delete args->executable;
  return nullptr;
}

PJRT_Error* Execute(PJRT_LoadedExecutable_Execute_Args* args) {
  g_last_arg_types.clear();
  g_last_arg_dims.clear();
  for (size_t i = 0; i < args->num_args; ++i) {
    g_last_arg_types.push_back(args->argument_lists[0][i]->type);
    g_last_arg_dims.push_back(args->argument_lists[0][i]->dims);
  }
  args->output_lists[0][0] = NewBuffer(PJRT_Buffer_Type_S64, {3});
  args->output_lists[0][1] = NewBuffer(PJRT_Buffer_Type_F32, {3, 2});
  args->output_lists[0][2] = NewBuffer(PJRT_Buffer_Type_U8, {3});
  if (args->device_complete_events) args->device_complete_events[0] = &g_event;
  return nullptr;
}

PJRT_Api MakeApi() {
  PJRT_Api api{};
  api.struct_size = PJRT_Api_STRUCT_SIZE;
  api.pjrt_api_version.major_version = PJRT_API_MAJOR;
  api.pjrt_api_version.minor_version = PJRT_API_MINOR;
  api.PJRT_Error_Destroy = ErrorDestroy;
  api.PJRT_Event_Destroy = EventDestroy;
  api.PJRT_Event_Await = EventAwait;
  api.PJRT_Client_Create = ClientCreate;
  api.PJRT_Client_Destroy = ClientDestroy;
  api.PJRT_Client_BufferFromHostBuffer = BufferFromHostBuffer;
  api.PJRT_Client_CreateViewOfDeviceBuffer = CreateView;
  api.PJRT_Client_Compile = Compile;
  api.PJRT_LoadedExecutable_Destroy = ExecutableDestroy;
  api.PJRT_LoadedExecutable_Execute = Execute;
  api.PJRT_Buffer_ElementType = ElementType;
  api.PJRT_Buffer_Dimensions = Dimensions;
  api.PJRT_Buffer_Destroy = Destroy;
  api.PJRT_Buffer_CopyToDevice = CopyToDevice;
  api.PJRT_Buffer_OpaqueDeviceMemoryDataPointer = DataPointer;
  api.PJRT_Buffer_Device = BufferDevice;
  api.PJRT_Buffer_ReadyEvent = ReadyEvent;
  return api;
}

PJRT_Api g_api = MakeApi();

}  // namespace

extern "C" __attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() { return &g_api; }

// Test hooks.
extern "C" __attribute__((visibility("default"))) const char* FakePlugin_LastProgram() {
  return g_last_program.c_str();
}
extern "C" __attribute__((visibility("default"))) int FakePlugin_LastArgType(size_t i) {
  return i < g_last_arg_types.size() ? g_last_arg_types[i] : -1;
}
extern "C" __attribute__((visibility("default"))) int64_t FakePlugin_LastArgDim(size_t i, size_t d) {
  return i < g_last_arg_dims.size() && d < g_last_arg_dims[i].size() ? g_last_arg_dims[i][d] : -1;
}
extern "C" __attribute__((visibility("default"))) PJRT_Buffer* FakePlugin_NewBuffer(int type, int64_t n) {
  return NewBuffer(static_cast<PJRT_Buffer_Type>(type), {n});
}
extern "C" __attribute__((visibility("default"))) PJRT_Device* FakePlugin_Device() { return &g_device; }
extern "C" __attribute__((visibility("default"))) int FakePlugin_ClientCreates() { return g_client_creates; }
extern "C" __attribute__((visibility("default"))) int FakePlugin_ClientDestroys() { return g_client_destroys; }
