CXX = g++
CXXFLAGS = -std=c++2a -Wall -fopenmp

SRCS_LIB := $(filter-out src/main_test.cpp, $(wildcard src/*.cpp))
OBJS_LIB := $(patsubst src/%.cpp,obj/%.o,$(SRCS_LIB))

all: test.exe

libourparallelFFT.a: $(OBJ_LIB)
	ar rs libourparallelFFT.a $^

test.exe: $(OBJS_LIB) obj/main_test.o
	$(CXX) $(LDFLAGS) -fopenmp $^ -o test.exe

obj/%.o : src/%.cpp | obj
	$(CXX) -c $(CXXFLAGS) -fopenmp $< -o $@

obj:
	mkdir -p obj

clear:
	rm -rf obj
	rm -f test.exe
	rm -f libourparallelFFT.a

.PHONY: all clear

