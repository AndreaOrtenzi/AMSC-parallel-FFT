CXX = g++
CXXFLAGS = -std=c++2a -Wall -fopenmp #-Wsuggest-override -Wnon-virtual-dtor
# LDFLAGS = -lfftw3 -lGetPot

SRCS_LIB := $(filter-out src/main_test.cpp, $(wildcard src/*.cpp))
OBJS_LIB := $(patsubst src/%.cpp,obj/%.o,$(SRCS_LIB))


all: test.exe
# all: lib

libourparallelFFT.a: $(OBJ_LIB)
	ar rs libourparallelFFT.a $^

# test our library
test.exe: $(OBJS_LIB) obj/main_test.o # tests/obj/main_test.o $(wildcard tests/obj/*.o)
	$(CXX) $(LDFLAGS) -fopenmp $^ -o test.exe
	
	
obj/%.o : src/%.cpp
	$(CXX) -c $(CXXFLAGS) -fopenmp  $^ -o $@
	
	
clear:
	rm -f obj/*.o
	rm -f test.exe
	rm -f libourparallelFFT.a

.PHONY: all clear
