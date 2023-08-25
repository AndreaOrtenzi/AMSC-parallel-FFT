CXX = g++
CXXFLAGS = -std=c++2a -Wall -fopenmp -I${mkEigenInc}

SRCS_LIB := $(filter-out src/main_test.cpp, $(wildcard src/*.cpp))
OBJS_LIB := $(patsubst src/%.cpp,obj/%.o,$(SRCS_LIB))

all: testDCT2D.exe

libourparallelDCT.a: $(OBJS_LIB)
	ar rs libourparallelDCT.a $^

testDCT2D.exe: $(OBJS_LIB) obj/main_test.o
	$(CXX) $(LDFLAGS) -fopenmp $^ -o testDCT2D.exe

obj/%.o : src/%.cpp | obj
	$(CXX) -c $(CXXFLAGS) -fopenmp $< -o $@

obj:
	mkdir -p obj

clear:
	rm -rf obj
	rm -f testDCT2D.exe
	rm -f libourparallelDCT.a

.PHONY: all clear


