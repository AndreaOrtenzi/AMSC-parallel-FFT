CXX = mpic++
CXXFLAGS = -std=c++2a -Wall -fopenmp
LDFLAGS = -fopenmp

SRCS_LIB := $(filter-out src/main_test.cpp, $(wildcard src/*.cpp))
OBJS_LIB := $(patsubst src/%.cpp,obj/%.o,$(SRCS_LIB))

all: test_library.exe test.exe

test_library.exe: lib/libourparallelFFT.a test_library.o
	$(CXX) $(LDFLAGS) test_library.o -L./lib -lourparallelFFT -o test_library.exe

test_library.o: test_library.cpp
	$(CXX) -c $(CXXFLAGS) -Iinc $< -o $@

lib/libourparallelFFT.a: $(OBJS_LIB)
	mkdir -p lib
	ar rvs lib/libourparallelFFT.a $^

test.exe: $(OBJS_LIB) obj/main_test.o
	$(CXX) $(LDFLAGS) $^ -o test.exe

obj/%.o : src/%.cpp | obj
	$(CXX) -c $(CXXFLAGS) $< -o $@

obj:
	mkdir -p obj

clear:
	rm -rf obj
	rm -rf lib
	rm -f *.o
	rm -f *.exe

.PHONY: all clear

