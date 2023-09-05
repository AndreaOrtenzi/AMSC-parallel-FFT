CXX = g++
CXXFLAGS = -std=c++2a -Wall -fopenmp -I${mkEigenInc} -Wno-deprecated
LDFLAGS = -fopenmp

SRCS_LIB := $(wildcard src/*.cpp)
OBJS_LIB := $(patsubst src/%.cpp,obj/%.o,$(SRCS_LIB))

all: main.exe

main.exe: $(OBJS_LIB)
	$(CXX) $(LDFLAGS) $^ -o main.exe

obj/%.o : src/%.cpp inc/parameters | obj
	$(CXX) -c $(CXXFLAGS) $< -o $@

obj:
	mkdir -p obj

clear:
	rm -rf obj
	rm -f main.exe
	rm -rf ./imgs/compressed/*

.PHONY: all clear


