CXX = g++-4.9
CXX_FLAGS = -std=c++11 -g -c  -I libsvm-3.20/ #-Wall

SOURCES += main.cpp

LIBS += -L/usr/local/lib/ -lopencv_features2d -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml -fopenmp -lopencv_nonfree
PROGRAM_NAME = SimpleClassifier

MAIN_OBJ_FILES = customsift.o main.o libsvm-3.20/svm.o codebook_2.o hog.o

bin: $(PROGRAM_NAME)

clean:
	- rm -f $(PROGRAM_NAME)
	- rm -f $(MAIN_OBJ_FILES)
	

SimpleClassifier: $(MAIN_OBJ_FILES)
	g++  $(MAIN_OBJ_FILES) -o $(PROGRAM_NAME) $(LIBS)
	
Alt: codebook_2.o new_main.o
	       $(CXX)  codebook_2.o new_main.o -o Alt $(LIBS)

main.o: main.cpp customsift.h libsvm-3.20/svm.h
	$(CXX) $(CXX_FLAGS) main.cpp -o main.o $(LIBS)
	
customsift.o: customsift.cpp customsift.h
	$(CXX) $(CXX_FLAGS) customsift.cpp $(LIBS) -o customsift.o
	
#codebook.o: codebook.cpp codebook.h
#	$(CXX) $(CXX_FLAGS) codebook.cpp $(LIBS) -o codebook.o
	
codebook_2.o: codebook_2.cpp codebook_2.h
	$(CXX) $(CXX_FLAGS) codebook_2.cpp $(LIBS) -o codebook_2.o
	
hog.o: hog.cpp hog.h
	$(CXX) $(CXX_FLAGS) hog.cpp $(LIBS) -o hog.o

svm.o: libsvm-3.20/svm.cpp libsvm-3.20/svm.h
	$(CXX) $(CXX_FLAGS) libsvm-3.20/svm.cpp


new_main.o: new_main.cpp
	        $(CXX) $(CXX_FLAGS) new_main.cpp -o new_main.o $(LIBS)
