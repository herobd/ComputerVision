CXX           = g++
CXXFLAGS      = -std=c++11 -g
LIBS          = -lopencv_viz -lopencv_core -lopencv_highgui -lopencv_calib3d -lopencv_imgproc
INCPATH       = -I.
TARGET        = TIP

make:
	$(CXX) $(CXXFLAGS) $(INCPATH) -o $(TARGET) main.cpp $(LIBS)

clean:
	rm $(TARGET)
