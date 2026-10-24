# Build the jax-metal PJRT shim and its tests.
#   make            -> build/pjrt_plugin_metal_shim.dylib
#   make test       -> run the C++ load tests (needs PLUGIN=path/to/pjrt_plugin_metal_14.dylib)
#   make probe      -> build/gpu_probe, the GPU-availability probe tests/guard.py uses

CXX ?= clang++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wno-missing-field-initializers
BUILD := build
FRAMEWORK_HEADERS := third_party/xla_0bfeeb47
PLUGIN_HEADERS := third_party/xla_3c8260f6
PLUGIN ?= ../re/targets/pjrt_plugin_metal_14-0.1.1.dylib
TEST_DATA ?= $(BUILD)/testdata

SHIM := $(BUILD)/pjrt_plugin_metal_shim.dylib

all: $(SHIM)

$(SHIM): src/pjrt_plugin_metal_shim.cc $(FRAMEWORK_HEADERS)/pjrt_c_api.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -fPIC -fvisibility=hidden -shared -I$(FRAMEWORK_HEADERS) \
	  -install_name @rpath/pjrt_plugin_metal_shim.dylib $< -o $@

$(BUILD)/direct_load: tests/direct_load.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) -Wno-unused-parameter -I$(PLUGIN_HEADERS) $< -o $@

$(BUILD)/shim_load: tests/shim_load.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) -Wno-unused-parameter -I$(FRAMEWORK_HEADERS) $< -o $@

$(BUILD)/gpu_probe: tests/gpu_probe.m | $(BUILD)
	clang -O2 -Wall -fobjc-arc -framework Foundation -framework Metal $< -o $@

probe: $(BUILD)/gpu_probe

$(BUILD):
	mkdir -p $@

test: $(SHIM) $(BUILD)/direct_load $(BUILD)/shim_load
	$(BUILD)/direct_load $(abspath $(PLUGIN)) $(TEST_DATA)
	JAX_METAL_SHIM_PLUGIN_PATH=$(abspath $(PLUGIN)) $(BUILD)/shim_load $(abspath $(SHIM)) $(TEST_DATA)

clean:
	rm -rf $(BUILD)

.PHONY: all test probe clean
