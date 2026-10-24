// Load Apple's pjrt_plugin_metal_14.dylib directly (no shim) through the PJRT C API
// 0.47, compile a StableHLO portable artifact and run it on the GPU.
//
//   direct_load PLUGIN_DYLIB TEST_DIR
//
// Built against the plugin's own header (third_party/xla_3c8260f6), which needs C++.

#include <dlfcn.h>

#include "pjrt_run.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s PLUGIN_DYLIB TEST_DIR\n", argv[0]);
    return 2;
  }
  void* plugin = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!plugin) {
    std::fprintf(stderr, "dlopen: %s\n", dlerror());
    return 2;
  }
  auto get_api = reinterpret_cast<const PJRT_Api* (*)()>(dlsym(plugin, "GetPjrtApi"));
  return pjrt_run::RunProgram(get_api(), argv[2]);
}
