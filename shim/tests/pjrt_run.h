// Compile and run a StableHLO portable artifact through a PJRT C API table and
// compare the result with the expected output written by gen_artifact.py.
//
// TEST_DIR holds program.mlirbc, compile_options.pb and manifest.txt. Every struct
// used here has the same layout in PJRT 0.47 and 0.116, so this compiles against
// either header.

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "pjrt_c_api.h"

namespace pjrt_run {

constexpr int kMaxTensors = 8;
constexpr int kMaxDims = 8;

struct Tensor {
  bool is_output;
  size_t num_dims;
  int64_t dims[kMaxDims];
  float* data;
  size_t count;
};

inline const PJRT_Api* api;

inline void* ReadFile(const char* path, size_t* size) {
  FILE* file = std::fopen(path, "rb");
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(2);
  }
  std::fseek(file, 0, SEEK_END);
  *size = static_cast<size_t>(std::ftell(file));
  std::fseek(file, 0, SEEK_SET);
  void* data = std::malloc(*size ? *size : 1);
  if (std::fread(data, 1, *size, file) != *size) {
    std::fprintf(stderr, "short read %s\n", path);
    std::exit(2);
  }
  std::fclose(file);
  return data;
}

// Print and abort on a PJRT error.
inline void Check(PJRT_Error* error, const char* what) {
  if (!error) return;
  PJRT_Error_Message_Args message = {.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE,
                                     .error = error};
  api->PJRT_Error_Message(&message);
  std::fprintf(stderr, "FAIL %s: %.*s\n", what, static_cast<int>(message.message_size),
               message.message);
  PJRT_Error_Destroy_Args destroy = {.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE,
                                     .error = error};
  api->PJRT_Error_Destroy(&destroy);
  std::exit(1);
}

inline void AwaitEvent(PJRT_Event* event, const char* what) {
  if (!event) return;
  PJRT_Event_Await_Args wait = {.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE, .event = event};
  Check(api->PJRT_Event_Await(&wait), what);
  PJRT_Event_Destroy_Args destroy = {.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE,
                                     .event = event};
  Check(api->PJRT_Event_Destroy(&destroy), "PJRT_Event_Destroy");
}

inline int ReadManifest(const char* dir, Tensor* tensors) {
  char path[4096];
  std::snprintf(path, sizeof path, "%s/manifest.txt", dir);
  FILE* file = std::fopen(path, "r");
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(2);
  }
  int count = 0;
  char role[8], dtype[8], data_path[4096];
  while (count < kMaxTensors &&
         std::fscanf(file, "%7s %7s %zu", role, dtype, &tensors[count].num_dims) == 3) {
    Tensor* tensor = &tensors[count];
    tensor->is_output = std::strcmp(role, "out") == 0;
    tensor->count = 1;
    for (size_t i = 0; i < tensor->num_dims; ++i) {
      if (std::fscanf(file, "%lld", reinterpret_cast<long long*>(&tensor->dims[i])) != 1) {
        std::exit(2);
      }
      tensor->count *= static_cast<size_t>(tensor->dims[i]);
    }
    if (std::fscanf(file, "%4095s", data_path) != 1 || std::strcmp(dtype, "f32") != 0) {
      std::exit(2);
    }
    size_t size;
    tensor->data = static_cast<float*>(ReadFile(data_path, &size));
    if (size != tensor->count * sizeof(float)) {
      std::fprintf(stderr, "size mismatch for %s\n", data_path);
      std::exit(2);
    }
    ++count;
  }
  std::fclose(file);
  return count;
}

// Returns 0 when the GPU result matches the expected output.
inline int RunProgram(const PJRT_Api* table, const char* dir) {
  api = table;
  std::printf("PJRT %d.%d, struct_size %zu\n", api->pjrt_api_version.major_version,
              api->pjrt_api_version.minor_version, api->struct_size);

  PJRT_Plugin_Initialize_Args init = {.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE};
  Check(api->PJRT_Plugin_Initialize(&init), "PJRT_Plugin_Initialize");

  PJRT_Client_Create_Args create = {.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE};
  Check(api->PJRT_Client_Create(&create), "PJRT_Client_Create");
  PJRT_Client* client = create.client;

  PJRT_Client_AddressableDevices_Args devices = {
      .struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE, .client = client};
  Check(api->PJRT_Client_AddressableDevices(&devices), "PJRT_Client_AddressableDevices");
  std::printf("addressable devices: %zu\n", devices.num_addressable_devices);
  PJRT_Device* device = devices.addressable_devices[0];

  char path[4096];
  size_t code_size, options_size;
  std::snprintf(path, sizeof path, "%s/program.mlirbc", dir);
  char* code = static_cast<char*>(ReadFile(path, &code_size));
  std::snprintf(path, sizeof path, "%s/compile_options.pb", dir);
  char* options = static_cast<char*>(ReadFile(path, &options_size));

  PJRT_Program program = {.struct_size = PJRT_Program_STRUCT_SIZE,
                          .code = code,
                          .code_size = code_size,
                          .format = "mlir",
                          .format_size = 4};
  PJRT_Client_Compile_Args compile = {.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
                                      .client = client,
                                      .program = &program,
                                      .compile_options = options,
                                      .compile_options_size = options_size};
  Check(api->PJRT_Client_Compile(&compile), "PJRT_Client_Compile");
  std::printf("compiled %zu-byte StableHLO artifact\n", code_size);

  Tensor tensors[kMaxTensors];
  int num_tensors = ReadManifest(dir, tensors);
  PJRT_Buffer* inputs[kMaxTensors];
  size_t num_inputs = 0;
  Tensor* expected = nullptr;
  for (int i = 0; i < num_tensors; ++i) {
    if (tensors[i].is_output) {
      expected = &tensors[i];
      continue;
    }
    PJRT_Client_BufferFromHostBuffer_Args upload = {
        .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
        .client = client,
        .data = tensors[i].data,
        .type = PJRT_Buffer_Type_F32,
        .dims = tensors[i].dims,
        .num_dims = tensors[i].num_dims,
        .host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,
        .device = device};
    Check(api->PJRT_Client_BufferFromHostBuffer(&upload), "PJRT_Client_BufferFromHostBuffer");
    AwaitEvent(upload.done_with_host_buffer, "done_with_host_buffer");
    inputs[num_inputs++] = upload.buffer;
  }

  PJRT_ExecuteOptions execute_options = {.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE};
  PJRT_Buffer* const* argument_lists[1] = {inputs};
  PJRT_Buffer* outputs[kMaxTensors] = {};
  PJRT_Buffer** output_lists[1] = {outputs};
  PJRT_Event* complete[1] = {};
  PJRT_LoadedExecutable_Execute_Args execute = {
      .struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
      .executable = compile.executable,
      .options = &execute_options,
      .argument_lists = argument_lists,
      .num_devices = 1,
      .num_args = num_inputs,
      .output_lists = output_lists,
      .device_complete_events = complete};
  Check(api->PJRT_LoadedExecutable_Execute(&execute), "PJRT_LoadedExecutable_Execute");
  AwaitEvent(complete[0], "device_complete_event");

  float* result = static_cast<float*>(std::malloc(expected->count * sizeof(float)));
  PJRT_Buffer_ToHostBuffer_Args download = {.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
                                            .src = outputs[0],
                                            .dst = result,
                                            .dst_size = expected->count * sizeof(float)};
  Check(api->PJRT_Buffer_ToHostBuffer(&download), "PJRT_Buffer_ToHostBuffer");
  AwaitEvent(download.event, "to_host_event");

  float max_error = 0;
  for (size_t i = 0; i < expected->count; ++i) {
    max_error = std::fmax(max_error, std::fabs(result[i] - expected->data[i]));
  }
  std::printf("result[0..3] = %g %g %g, expected %g %g %g\n", result[0], result[1], result[2],
              expected->data[0], expected->data[1], expected->data[2]);
  std::printf("max abs error vs CPU: %g\n", max_error);
  if (max_error > 1e-3f) {
    std::printf("FAIL: result mismatch\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}

}  // namespace pjrt_run
