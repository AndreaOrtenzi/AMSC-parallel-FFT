CXX = g++
CXXFLAGS = -std=c++2a -Wall -fopenmp
LDFLAGS = -fopenmp

# SRCS_LIB := $(filter-out src/main_test.cpp, $(wildcard src/*.cpp))
# OBJS_LIB := $(patsubst src/%.cpp,obj/%.o,$(SRCS_LIB))
SRCS_LIB := $(wildcard src/*.cpp)
OBJS_LIB := $(patsubst src/%.cpp,obj/%.o,$(SRCS_LIB))

all: test.exe

test.exe: $(OBJS_LIB)
	$(CXX) $(LDFLAGS) $^ -o test.exe

obj/%.o : src/%.cpp | obj
	$(CXX) -c $(CXXFLAGS) $< -o $@

obj:
	mkdir -p obj

clear:
	rm -rf obj
	rm -f test.exe

.PHONY: all clear


