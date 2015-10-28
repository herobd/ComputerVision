CXX = g++-4.9
CXX_FLAGS = -std=c++11 -g -c  -I libsvm-3.20/ #-Wall

SOURCES += main.cpp

LIBS += -L/urs/local/include/opencv2 -lopencv_features2d -lopencv_core -lopencv_highgui -lopencv_nonfree -lopencv_imgproc -fopenmp
PROGRAM_NAME = SimpleClassifier

MAIN_OBJ_FILES = customsift.o main.o libsvm-3.20/svm.o codebook.o

bin: $(PROGRAM_NAME)

clean:
	- rm -f $(PROGRAM_NAME)
	- rm -f $(MAIN_OBJ_FILES)
	

SimpleClassifier: $(MAIN_OBJ_FILES)
	g++  $(MAIN_OBJ_FILES) -o $(PROGRAM_NAME) $(LIBS)
	
main.o: main.cpp customsift.h libsvm-3.20/svm.h
	$(CXX) $(CXX_FLAGS) main.cpp -o main.o $(LIBS)
	
customsift.o: customsift.cpp customsift.h
	$(CXX) $(CXX_FLAGS) customsift.cpp $(LIBS) -o customsift.o
	
codebook.o: codebook.cpp codebook.h
	$(CXX) $(CXX_FLAGS) codebook.cpp $(LIBS) -o codebook.o
	
svm.o: libsvm-3.20/svm.cpp libsvm-3.20/svm.h
	$(CXX) $(CXX_FLAGS) libsvm-3.20/svm.cpp
