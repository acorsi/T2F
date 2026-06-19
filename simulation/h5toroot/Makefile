# Compiler
CXX = g++

# Flags de compilation
CXXFLAGS = -std=c++17 -Wall -O3

# HDF5
HDF5_CFLAGS = -I$(CONDA_PREFIX)/include
HDF5_LIBS = -L$(CONDA_PREFIX)/lib -lhdf5_cpp -lhdf5

# ROOT
ROOT_CFLAGS = $(shell root-config --cflags)
ROOT_LIBS = $(shell root-config --libs)

# Cible principale
TARGET = h5toroot

# Sources
SRCS = convert.cpp
OBJS = $(SRCS:.cpp=.o)

# Règles
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(HDF5_LIBS) $(ROOT_LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(HDF5_CFLAGS) $(ROOT_CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

install: $(TARGET)
	cp $(TARGET) $(HOME)/bin/

.PHONY: all clean install
