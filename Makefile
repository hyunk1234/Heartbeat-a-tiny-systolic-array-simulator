# Builds the C++ core as a shared library (used by Python via ctypes and by
# the C example), plus the C++ unit tests and benchmark.
#
#   make          build everything into build/
#   make test     run C++ tests, the C example, and the Python tests
#   make bench    run the C++ benchmark
#   make clean

CXX      ?= c++
CC       ?= cc
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pedantic
CFLAGS   ?= -O2 -std=c99 -Wall -Wextra -pedantic
PYTHON   ?= python3

BUILD := build
INC   := -Icpp/include
SRC   := cpp/src/common.cpp cpp/src/output_stationary.cpp cpp/src/weight_stationary.cpp cpp/src/c_api.cpp
HDR   := cpp/include/heartbeat.hpp cpp/include/heartbeat.h

ifeq ($(shell uname -s),Darwin)
  LIB    := $(BUILD)/libheartbeat.dylib
  SHARED := -dynamiclib -install_name @rpath/libheartbeat.dylib
  RPATH  := -Wl,-rpath,@loader_path
else
  LIB    := $(BUILD)/libheartbeat.so
  SHARED := -shared
  RPATH  := -Wl,-rpath,'$$ORIGIN'
endif

.PHONY: all test bench clean

all: $(LIB) $(BUILD)/hb_tests $(BUILD)/hb_bench $(BUILD)/c_example

$(BUILD):
	mkdir -p $@

$(LIB): $(SRC) $(HDR) | $(BUILD)
	$(CXX) $(CXXFLAGS) -fPIC -fvisibility=hidden $(INC) $(SHARED) $(SRC) -o $@

$(BUILD)/hb_tests: cpp/tests/test_main.cpp $(SRC) $(HDR) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) cpp/tests/test_main.cpp $(SRC) -o $@

$(BUILD)/hb_bench: cpp/tools/hb_bench.cpp $(SRC) $(HDR) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) cpp/tools/hb_bench.cpp $(SRC) -o $@

$(BUILD)/c_example: examples/c_example.c $(LIB) | $(BUILD)
	$(CC) $(CFLAGS) $(INC) $< -L$(BUILD) -lheartbeat $(RPATH) -o $@

test: all
	./$(BUILD)/hb_tests
	./$(BUILD)/c_example > /dev/null
	$(PYTHON) -m unittest discover -s tests

bench: $(BUILD)/hb_bench
	./$(BUILD)/hb_bench --m 128 --k 128 --n 128 --rows 32 --cols 32

clean:
	rm -rf $(BUILD)
