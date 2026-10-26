// Tests the shim's buffer retyping and boundary programs against fake_pjrt_plugin.cc, on any
// machine (no GPU, no Apple plugin):
//
//   make -C shim unit-test
//
// uint64 buffers must live in the plugin as int64 and be reported as uint64; the program the
// plugin compiles must be the boundary program; at execution complex64 arguments must reach
// the plugin as float32[..., 2] views, stray uint64 buffers as int64 views, and outputs at
// boundary positions must come back as uint64 and complex64.

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

  // Copies keep the logical type.
  PJRT_Buffer_CopyToDevice_Args copy{};
  copy.struct_size = PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE;
  copy.buffer = u64;
  copy.dst_device = device;
  CHECK(g_api->PJRT_Buffer_CopyToDevice(&copy) == nullptr);
  CHECK(TypeOf(copy.dst_buffer) == PJRT_Buffer_Type_U64);
  Destroy(copy.dst_buffer);

  // The plugin compiles the boundary program handed over for this compilation.
  const int32_t in_positions[] = {0, 1}, in_kinds[] = {2, 1};
  const int32_t out_positions[] = {0, 1}, out_kinds[] = {2, 1};
  const char boundary[] = "boundary program";
  set_io(in_positions, in_kinds, 2, out_positions, out_kinds, 2, boundary, sizeof(boundary) - 1);
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
  // the outputs come back as uint64[3] and complex64[3].
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
  PJRT_Buffer* outputs[2] = {};
  execute(u64, c64, outputs);
  CHECK(last_arg_type(0) == PJRT_Buffer_Type_S64);
  CHECK(last_arg_type(1) == PJRT_Buffer_Type_F32);
  CHECK(last_arg_dim(1, 0) == 3 && last_arg_dim(1, 1) == 2);
  CHECK(TypeOf(outputs[0]) == PJRT_Buffer_Type_U64);
  CHECK(DimsOf(outputs[0]) == "3");
  CHECK(TypeOf(outputs[1]) == PJRT_Buffer_Type_C64);
  CHECK(DimsOf(outputs[1]) == "3");
  CHECK(TypeOf(c64) == PJRT_Buffer_Type_C64);  // the argument itself is unchanged
  Destroy(outputs[0]);
  Destroy(outputs[1]);

  // A uint64 buffer the plugin holds as uint64 (not made through the shim) goes in as an
  // int64 view.
  PJRT_Buffer* stray = new_buffer(PJRT_Buffer_Type_U64, 3);
  execute(stray, c64, outputs);
  CHECK(last_arg_type(0) == PJRT_Buffer_Type_S64);
  Destroy(outputs[0]);
  Destroy(outputs[1]);
  Destroy(stray);

  // Destroyed buffers are forgotten: a new buffer at a reused address is not misreported.
  Destroy(u64);
  PJRT_Buffer* fresh = FromHost(ints, PJRT_Buffer_Type_S64, 3, device);
  CHECK(TypeOf(fresh) == PJRT_Buffer_Type_S64);
  Destroy(fresh);
  Destroy(s64);
  Destroy(c64);

  PJRT_LoadedExecutable_Destroy_Args drop{};
  drop.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  drop.executable = compile.executable;
  CHECK(g_api->PJRT_LoadedExecutable_Destroy(&drop) == nullptr);

  std::printf(g_failures ? "shim_boundary_test: %d FAILED\n" : "shim_boundary_test: all passed\n",
              g_failures);
  return g_failures ? 1 : 0;
}
