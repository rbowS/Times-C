CC = nvcc
I_DIR = -I./include 
BIN = ./bin/test
#test normalization
#SRC = ./src/normalization.cu ./src/readfile.cu
#SRC = ./src/test_norm.cu 
#test ncc
#SRC = ./src/ncc.cu ./test/test_ncc.cu

#test SBD
#SRC = ./src/SBD.cu ./test/test_sbd.cu


#test shapeExtract
#SRC = ./src/shapeExtract.cu ./test/test_shapeExtra.cu


#test kshape
SRC = ./test/test_kshape.cu

CLEAN = clean
COMPUTE = sm_70
RDC = true
MATRIX_LIB = cublas
FFT_LIB = cufft
EIGEN_LIB = cusolver
RAND_LIB = curand
$(BIN):$(SRC)
	$(CC) -std=c++11 $^ -o $@ -l$(MATRIX_LIB) -l$(FFT_LIB) -l$(EIGEN_LIB) -l$(RAND_LIB) -arch=$(COMPUTE) -rdc=$(RDC) -lcudadevrt $(I_DIR)

.PHONY:$(CLEAN)
$(CLEAN):
	rm -rf $(BIN)

