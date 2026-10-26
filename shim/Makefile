# Build the jax-metal PJRT shim and its tests.
#   make            -> build/pjrt_plugin_metal_shim.dylib (for the C++ tests; pip builds its own)
#   make test       -> run the C++ load tests (needs PLUGIN=path/to/pjrt_plugin_metal_14.dylib)
#   make probe      -> build/gpu_probe, the GPU-availability probe tests/guard.py uses
#   make gpu-session -> every GPU test, resumable (tests/gpu_session.py; ARGS="--dry-run" etc.)
#   make unit-test  -> the shim against a fake plugin (tests/shim_boundary_test.cc); any machine

CXX ?= clang++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wno-missing-field-initializers
BUILD := build
FRAMEWORK_HEADERS := third_party/xla_0bfeeb47
PLUGIN_HEADERS := third_party/xla_3c8260f6
PLUGIN ?= ../re/targets/pjrt_plugin_metal_14-0.1.1.dylib
TEST_DATA ?= $(BUILD)/testdata
VENV_PYTHON ?= $(firstword $(wildcard ../.venv-shim/bin/python) python3)
VERSION := $(shell sed -n 's/^__version__ = "\(.*\)"/\1/p' python/jax_plugins/metal_shim/__init__.py)

SHIM := $(BUILD)/pjrt_plugin_metal_shim.dylib

all: $(SHIM)

$(SHIM): src/pjrt_plugin_metal_shim.cc $(FRAMEWORK_HEADERS)/pjrt_c_api.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -fPIC -fvisibility=hidden -shared -I$(FRAMEWORK_HEADERS) \
	  -DJAX_METAL_SHIM_VERSION='"$(VERSION)"' \
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

UNIT := $(BUILD)/unit
LDL := $(if $(filter Linux,$(shell uname)),-ldl)

unit-test: | $(BUILD)
	mkdir -p $(UNIT)
	$(CXX) $(CXXFLAGS) -fPIC -fvisibility=hidden -shared -I$(FRAMEWORK_HEADERS) \
	  -DJAX_METAL_SHIM_VERSION='"$(VERSION)"' src/pjrt_plugin_metal_shim.cc -o $(UNIT)/shim.so
	$(CXX) $(CXXFLAGS) -fPIC -fvisibility=hidden -shared -I$(PLUGIN_HEADERS) \
	  tests/fake_pjrt_plugin.cc -o $(UNIT)/fake_plugin.so
	$(CXX) $(CXXFLAGS) -I$(FRAMEWORK_HEADERS) tests/shim_boundary_test.cc \
	  -o $(UNIT)/shim_boundary_test $(LDL)
	$(UNIT)/shim_boundary_test $(abspath $(UNIT)/shim.so) $(abspath $(UNIT)/fake_plugin.so)

gpu-session:
	$(VENV_PYTHON) tests/gpu_session.py $(ARGS)

clean:
	rm -rf $(BUILD)

.PHONY: all test probe unit-test gpu-session clean
