CXX = g++
CXXFLAGS = -std=c++2a -Wall -fopenmp -I${mkEigenInc}

SRCS_LIB := $(filter-out src/main_test.cpp, $(wildcard src/*.cpp))
OBJS_LIB := $(patsubst src/%.cpp,obj/%.o,$(SRCS_LIB))

all: test2D.exe

test2D.exe: $(OBJS_LIB) obj/main_test.o
	$(CXX) $(LDFLAGS) -fopenmp $^ -o test2D.exe

obj/%.o : src/%.cpp | obj
	$(CXX) -c $(CXXFLAGS) -fopenmp $< -o $@

obj:
	mkdir -p obj

clear:
	rm -rf obj
	rm -f test2D.exe
	rm -f libourparallelFFT.a

.PHONY: all clear


