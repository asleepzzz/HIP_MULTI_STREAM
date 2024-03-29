HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

SOURCES = test2.cpp test_common.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./test_multi

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g
CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@


test: $(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

