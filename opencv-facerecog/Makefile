
CXX=g++
CV_FLAGS=`pkg-config --cflags opencv`
CV_LIBS=`pkg-config --libs opencv`
LIBS=-lm -lboost_filesystem -lboost_system
FLAGS=-O3 -Wall -funroll-loops
RM=rm -f

all: opencv-facerecog

opencv-facerecog: opencv-facerecog.o
	$(CXX) $(FLAGS) $(CV_FLAGS) opencv-facerecog.o -o opencv-facerecog $(CV_LIBS) $(LIBS)

opencv-facerecog.o: opencv-facerecog.cpp
	$(CXX) $(FLAGS) $(CV_FLAGS) -c opencv-facerecog.cpp $(CV_LIBS) $(LIBS)

clean:
	$(RM) opencv-facerecog.o opencv-facerecog *~

tidy:
	$(RM) opencv-facerecog.o *~


