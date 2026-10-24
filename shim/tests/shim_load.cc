// Load pjrt_plugin_metal_shim.dylib, check what it reports to the framework, then
// compile and run a StableHLO portable artifact through it on the GPU.
//
//   JAX_METAL_SHIM_PLUGIN_PATH=PLUGIN_DYLIB shim_load SHIM_DYLIB TEST_DIR
//
// Built against the framework-side header (third_party/xla_0bfeeb47, PJRT 0.116).

#include <dlfcn.h>

#include <cstdint>
#include <string_view>

#include "pjrt_run.h"

namespace {

int failures = 0;

void Expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  failures += !ok;
}

bool HasVersion(const PJRT_NamedValue* attributes, size_t count, std::string_view name,
                int64_t major, int64_t minor, int64_t patch) {
  for (size_t i = 0; i < count; ++i) {
    const PJRT_NamedValue& value = attributes[i];
    if (std::string_view(value.name, value.name_size) == name &&
        value.type == PJRT_NamedValue_kInt64List && value.value_size == 3) {
      return value.int64_array_value[0] == major && value.int64_array_value[1] == minor &&
             value.int64_array_value[2] == patch;
    }
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s SHIM_DYLIB TEST_DIR\n", argv[0]);
    return 2;
  }
  void* shim = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!shim) {
    std::fprintf(stderr, "dlopen: %s\n", dlerror());
    return 2;
  }
  auto get_api = reinterpret_cast<const PJRT_Api* (*)()>(dlsym(shim, "GetPjrtApi"));
  const PJRT_Api* api = get_api ? get_api() : nullptr;
  Expect(api != nullptr, "GetPjrtApi returns a table");
  if (!api) return 1;

  Expect(api->pjrt_api_version.major_version == 0 && api->pjrt_api_version.minor_version == 47,
         "reports the plugin's PJRT version 0.47");
  Expect(api->struct_size == 808, "reports the plugin's struct_size 808");

  PJRT_Plugin_Attributes_Args attributes = {.struct_size = PJRT_Plugin_Attributes_Args_STRUCT_SIZE};
  Expect(api->PJRT_Plugin_Attributes(&attributes) == nullptr, "PJRT_Plugin_Attributes succeeds");
  Expect(HasVersion(attributes.attributes, attributes.num_attributes,
                    "stablehlo_current_version", 0, 19, 4),
         "stablehlo_current_version = 0.19.4");
  Expect(HasVersion(attributes.attributes, attributes.num_attributes,
                    "stablehlo_minimum_version", 0, 9, 0),
         "stablehlo_minimum_version = 0.9.0");

  // Entry points jaxlib reads with only a null check must be null past the plugin's table.
  Expect(api->PJRT_Memory_Kind_Id == nullptr, "first entry past the plugin table is null");
  Expect(api->PJRT_Client_Load == nullptr, "PJRT_Client_Load is null");
  Expect(api->PJRT_LoadedExecutable_AddressableDeviceLogicalIds == nullptr,
         "PJRT_LoadedExecutable_AddressableDeviceLogicalIds is null");
  Expect(api->PJRT_Event_Create == nullptr && api->PJRT_Event_Set == nullptr,
         "PJRT_Event_Create/Set are null");
  Expect(api->PJRT_TopologyDescription_GetMemorySpaceKindIds == nullptr,
         "last 0.116 entry is null");
  auto* raw = reinterpret_cast<void* const*>(api);
  bool padded_null = true;
  for (size_t i = sizeof(PJRT_Api) / sizeof(void*); i < 512; ++i) padded_null &= raw[i] == nullptr;
  Expect(padded_null, "padding past the 0.116 table is null");
  Expect(api->PJRT_Client_Compile != nullptr && api->PJRT_LoadedExecutable_Execute != nullptr,
         "plugin entries are forwarded");

  if (failures) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  return pjrt_run::RunProgram(api, argv[2]);
}
