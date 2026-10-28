// Tests the shim's buffer retyping and boundary programs against fake_pjrt_plugin.cc, on any
// machine (no GPU, no Apple plugin):
//
//   make -C shim unit-test
//
// uint64 buffers must live in the plugin as int64, float8 and narrow-integer ones as uint8 or
// int8, and be reported with their own type; the program the plugin compiles must be the
// boundary program; at execution complex64 arguments must reach the plugin as
// float32[..., 2] views, stray uint64 buffers as int64 views, and outputs at boundary
// positions must come back as uint64, complex64 and float8_e4m3fn.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "pjrt_c_api.h"  // PJRT 0.116, the framework-side layout.

namespace {

int g_failures = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

const PJRT_Api* g_api;

PJRT_Buffer_Type TypeOf(PJRT_Buffer* buffer) {
  PJRT_Buffer_ElementType_Args args{};
  args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  args.buffer = buffer;
  CHECK(g_api->PJRT_Buffer_ElementType(&args) == nullptr);
  return args.type;
}

std::string DimsOf(PJRT_Buffer* buffer) {
  PJRT_Buffer_Dimensions_Args args{};
  args.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  args.buffer = buffer;
  CHECK(g_api->PJRT_Buffer_Dimensions(&args) == nullptr);
  std::string text;
  for (size_t i = 0; i < args.num_dims; ++i) text += (i ? "," : "") + std::to_string(args.dims[i]);
  return text;
}

PJRT_Buffer* FromHost(const void* data, PJRT_Buffer_Type type, int64_t n, PJRT_Device* device) {
  PJRT_Client_BufferFromHostBuffer_Args args{};
  args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  args.data = data;
  args.type = type;
  args.dims = &n;
  args.num_dims = 1;
  args.device = device;
  CHECK(g_api->PJRT_Client_BufferFromHostBuffer(&args) == nullptr);
  CHECK(args.type == type);  // the caller's arguments are restored
  return args.buffer;
}

void Destroy(PJRT_Buffer* buffer) {
  PJRT_Buffer_Destroy_Args args{};
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.buffer = buffer;
  CHECK(g_api->PJRT_Buffer_Destroy(&args) == nullptr);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s SHIM_LIBRARY FAKE_PLUGIN_LIBRARY\n", argv[0]);
    return 2;
  }
  setenv("JAX_METAL_SHIM_PLUGIN_PATH", argv[2], 1);
  void* shim = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  void* fake = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (!shim || !fake) {
    std::fprintf(stderr, "cannot load: %s\n", dlerror());
    return 2;
  }
  auto get_api = reinterpret_cast<const PJRT_Api* (*)()>(dlsym(shim, "GetPjrtApi"));
  auto set_io = reinterpret_cast<void (*)(const int32_t*, const int32_t*, size_t, const int32_t*,
                                          const int32_t*, size_t, const char*, size_t)>(
      dlsym(shim, "JaxMetalShim_SetBoundaryIo"));
  auto last_program = reinterpret_cast<const char* (*)()>(dlsym(fake, "FakePlugin_LastProgram"));
  auto last_arg_type = reinterpret_cast<int (*)(size_t)>(dlsym(fake, "FakePlugin_LastArgType"));
  auto last_arg_dim = reinterpret_cast<int64_t (*)(size_t, size_t)>(dlsym(fake, "FakePlugin_LastArgDim"));
  auto new_buffer = reinterpret_cast<PJRT_Buffer* (*)(int, int64_t)>(dlsym(fake, "FakePlugin_NewBuffer"));
  auto fake_device = reinterpret_cast<PJRT_Device* (*)()>(dlsym(fake, "FakePlugin_Device"));
  auto client_creates = reinterpret_cast<int (*)()>(dlsym(fake, "FakePlugin_ClientCreates"));
  auto client_destroys = reinterpret_cast<int (*)()>(dlsym(fake, "FakePlugin_ClientDestroys"));
  g_api = get_api ? get_api() : nullptr;
  if (!g_api || !set_io) {
    std::fprintf(stderr, "the shim did not load the fake plugin\n");
    return 2;
  }
  auto plugin_api = reinterpret_cast<const PJRT_Api* (*)()>(dlsym(fake, "GetPjrtApi"))();
  auto plugin_type = [&](PJRT_Buffer* buffer) {
    PJRT_Buffer_ElementType_Args args{};
    args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
    args.buffer = buffer;
    CHECK(plugin_api->PJRT_Buffer_ElementType(&args) == nullptr);
    return args.type;
  };
  PJRT_Device* device = fake_device();

  // Repeated framework lifetimes share one native client. Destroying one
  // framework client cannot invalidate another or allocate a client per reset.
  PJRT_Client* retained = nullptr;
  for (int i = 0; i < 5; ++i) {
    PJRT_Client_Create_Args create{};
    create.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    CHECK(g_api->PJRT_Client_Create(&create) == nullptr);
    if (!retained) retained = create.client;
    CHECK(create.client == retained);
    PJRT_Client_Destroy_Args destroy{};
    destroy.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    destroy.client = create.client;
    CHECK(g_api->PJRT_Client_Destroy(&destroy) == nullptr);
  }
  CHECK(client_creates() == 1);
  CHECK(client_destroys() == 0);  // release is deferred to process exit

  // A missing optional API returns a readable UNIMPLEMENTED error, never a
  // null function pointer or fabricated memory statistics. Repeated requests
  // and destruction must leave the error valid for another caller.
  for (int i = 0; i < 3; ++i) {
    PJRT_Executable_GetCompiledMemoryStats_Args stats{};
    stats.struct_size = PJRT_Executable_GetCompiledMemoryStats_Args_STRUCT_SIZE;
    CHECK(g_api->PJRT_Executable_GetCompiledMemoryStats != nullptr);
    PJRT_Error* error = g_api->PJRT_Executable_GetCompiledMemoryStats(&stats);
    CHECK(error != nullptr);
    PJRT_Error_GetCode_Args code{};
    code.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
    code.error = error;
    CHECK(g_api->PJRT_Error_GetCode(&code) == nullptr);
    CHECK(code.code == PJRT_Error_Code_UNIMPLEMENTED);
    PJRT_Error_Message_Args message{};
    message.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    message.error = error;
    g_api->PJRT_Error_Message(&message);
    CHECK(std::string(message.message, message.message_size).find("memory statistics") !=
          std::string::npos);
    PJRT_Error_Destroy_Args drop_error{};
    drop_error.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    drop_error.error = error;
    g_api->PJRT_Error_Destroy(&drop_error);
  }

  // uint64 lives in the plugin as int64 and is reported as uint64; other types are untouched.
  const uint64_t words[3] = {1, 1ull << 63, ~0ull};
  PJRT_Buffer* u64 = FromHost(words, PJRT_Buffer_Type_U64, 3, device);
  CHECK(plugin_type(u64) == PJRT_Buffer_Type_S64);
  CHECK(TypeOf(u64) == PJRT_Buffer_Type_U64);
  const int64_t ints[3] = {1, 2, 3};
  PJRT_Buffer* s64 = FromHost(ints, PJRT_Buffer_Type_S64, 3, device);
  CHECK(TypeOf(s64) == PJRT_Buffer_Type_S64);
  const float pairs[6] = {1, 2, 3, 4, 5, 6};
  PJRT_Buffer* c64 = FromHost(pairs, PJRT_Buffer_Type_C64, 3, device);
  CHECK(TypeOf(c64) == PJRT_Buffer_Type_C64);

  // Narrow types: one value per byte, held as uint8 (int8 for the signed integers).
  const unsigned char bytes[3] = {0x38, 0x0F, 0x7F};
  const struct {
    PJRT_Buffer_Type type, physical;
  } narrow[] = {{PJRT_Buffer_Type_F8E4M3FN, PJRT_Buffer_Type_U8},
                {PJRT_Buffer_Type_F8E5M2FNUZ, PJRT_Buffer_Type_U8},
                {PJRT_Buffer_Type_F8E8M0FNU, PJRT_Buffer_Type_U8},
                {PJRT_Buffer_Type_F4E2M1FN, PJRT_Buffer_Type_U8},
                {PJRT_Buffer_Type_S4, PJRT_Buffer_Type_S8},
                {PJRT_Buffer_Type_U4, PJRT_Buffer_Type_U8},
                {PJRT_Buffer_Type_S2, PJRT_Buffer_Type_S8}};
  for (const auto& entry : narrow) {
    PJRT_Buffer* buffer = FromHost(bytes, entry.type, 3, device);
    CHECK(plugin_type(buffer) == entry.physical);
    CHECK(TypeOf(buffer) == entry.type);
    Destroy(buffer);
  }

  // Copies keep the logical type.
  PJRT_Buffer_CopyToDevice_Args copy{};
  copy.struct_size = PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE;
  copy.buffer = u64;
  copy.dst_device = device;
  CHECK(g_api->PJRT_Buffer_CopyToDevice(&copy) == nullptr);
  CHECK(TypeOf(copy.dst_buffer) == PJRT_Buffer_Type_U64);
  Destroy(copy.dst_buffer);

  // The plugin compiles the boundary program handed over for this compilation. Kinds:
  // 0x100 | type for retyped types (here uint64 and float8_e4m3fn), 1 for complex pairs.
  const int32_t in_positions[] = {0, 1}, in_kinds[] = {0x100 | PJRT_Buffer_Type_U64, 1};
  const int32_t out_positions[] = {0, 1, 2};
  const int32_t out_kinds[] = {0x100 | PJRT_Buffer_Type_U64, 1, 0x100 | PJRT_Buffer_Type_F8E4M3FN};
  const char boundary[] = "boundary program";
  set_io(in_positions, in_kinds, 2, out_positions, out_kinds, 3, boundary, sizeof(boundary) - 1);
  char original[] = "original program";
  PJRT_Program program{};
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.code = original;
  program.code_size = sizeof(original) - 1;
  PJRT_Client_Compile_Args compile{};
  compile.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  compile.program = &program;
  CHECK(g_api->PJRT_Client_Compile(&compile) == nullptr);
  set_io(nullptr, nullptr, 0, nullptr, nullptr, 0, nullptr, 0);
  CHECK(std::strcmp(last_program(), boundary) == 0);

  // Execution: complex64 goes in as a float32[3, 2] view, the int64-held uint64 as is, and
  // the outputs come back as uint64[3], complex64[3] and float8_e4m3fn[3].
  auto execute = [&](PJRT_Buffer* first, PJRT_Buffer* second, PJRT_Buffer** outputs) {
    PJRT_Buffer* arguments[] = {first, second};
    PJRT_Buffer* const* argument_lists[] = {arguments};
    PJRT_Buffer** output_lists[] = {outputs};
    PJRT_LoadedExecutable_Execute_Args run{};
    run.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    run.executable = compile.executable;
    run.argument_lists = argument_lists;
    run.num_devices = 1;
    run.num_args = 2;
    run.output_lists = output_lists;
    CHECK(g_api->PJRT_LoadedExecutable_Execute(&run) == nullptr);
  };
  PJRT_Buffer* outputs[3] = {};
  execute(u64, c64, outputs);
  CHECK(last_arg_type(0) == PJRT_Buffer_Type_S64);
  CHECK(last_arg_type(1) == PJRT_Buffer_Type_F32);
  CHECK(last_arg_dim(1, 0) == 3 && last_arg_dim(1, 1) == 2);
  CHECK(TypeOf(outputs[0]) == PJRT_Buffer_Type_U64);
  CHECK(DimsOf(outputs[0]) == "3");
  CHECK(TypeOf(outputs[1]) == PJRT_Buffer_Type_C64);
  CHECK(DimsOf(outputs[1]) == "3");
  CHECK(TypeOf(outputs[2]) == PJRT_Buffer_Type_F8E4M3FN);
  CHECK(plugin_type(outputs[2]) == PJRT_Buffer_Type_U8);
  CHECK(TypeOf(c64) == PJRT_Buffer_Type_C64);  // the argument itself is unchanged
  for (PJRT_Buffer* output : outputs) Destroy(output);

  // A uint64 buffer the plugin holds as uint64 (not made through the shim) goes in as an
  // int64 view.
  PJRT_Buffer* stray = new_buffer(PJRT_Buffer_Type_U64, 3);
  execute(stray, c64, outputs);
  CHECK(last_arg_type(0) == PJRT_Buffer_Type_S64);
  for (PJRT_Buffer* output : outputs) Destroy(output);
  Destroy(stray);

  PJRT_LoadedExecutable_Destroy_Args drop{};
  drop.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  drop.executable = compile.executable;
  CHECK(g_api->PJRT_LoadedExecutable_Destroy(&drop) == nullptr);

  // Empty input buffers keep their logical metadata, while the native program
  // receives a nonempty scalar placeholder in the same argument position.
  const int32_t empty_position = 0, empty_kind = 2;
  set_io(&empty_position, &empty_kind, 1, nullptr, nullptr, 0,
         boundary, sizeof(boundary) - 1);
  CHECK(g_api->PJRT_Client_Compile(&compile) == nullptr);
  set_io(nullptr, nullptr, 0, nullptr, nullptr, 0, nullptr, 0);
  PJRT_Buffer* empty = FromHost(ints, PJRT_Buffer_Type_S32, 0, device);
  execute(empty, s64, outputs);
  CHECK(last_arg_type(0) == PJRT_Buffer_Type_U8);
  CHECK(last_arg_dim(0, 0) == -1);  // scalar
  CHECK(TypeOf(empty) == PJRT_Buffer_Type_S32);
  CHECK(DimsOf(empty) == "0");
  for (PJRT_Buffer* output : outputs) Destroy(output);
  Destroy(empty);

  // Destroyed buffers are forgotten: a new buffer at a reused address is not misreported.
  Destroy(u64);
  PJRT_Buffer* fresh = FromHost(ints, PJRT_Buffer_Type_S64, 3, device);
  CHECK(TypeOf(fresh) == PJRT_Buffer_Type_S64);
  Destroy(fresh);
  Destroy(s64);
  Destroy(c64);

  drop.executable = compile.executable;
  CHECK(g_api->PJRT_LoadedExecutable_Destroy(&drop) == nullptr);

  std::printf(g_failures ? "shim_boundary_test: %d FAILED\n" : "shim_boundary_test: all passed\n",
              g_failures);
  return g_failures ? 1 : 0;
}
