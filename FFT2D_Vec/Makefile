CXX = g++
CXXFLAGS = -std=c++2a -Wall -fopenmp
LDFLAGS = -fopenmp

SRCS_LIB := $(wildcard src/*.cpp)
TEMPLATE_LIB := $(wildcard src/*.tpp)
OBJS_LIB := $(patsubst src/%.cpp,obj/%.o,$(SRCS_LIB))

all: test2D.exe

test2D.exe: $(OBJS_LIB)
	$(CXX) $(LDFLAGS) $^ -o test2D.exe

obj/%.o : src/%.cpp $(TEMPLATE_LIB) inc/parameters | obj
	$(CXX) -c $(CXXFLAGS) $< -o $@

obj:
	mkdir -p obj

clear:
	rm -rf obj
	rm -f test2D.exe

.PHONY: all clear


