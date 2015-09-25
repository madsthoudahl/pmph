CXX        = nvcc

SOURCES_CPP =test.cu
HELPERS     =hostlib.cu.h devlib.cu.h devkernels.cu.h
EXECUTABLE  =test


default: compile

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)
$(EXECUTABLE): 
	$(CXX) -o $(EXECUTABLE) $(SOURCES_CPP)


run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)

