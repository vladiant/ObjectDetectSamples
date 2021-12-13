# Default OpenCV include dir: /usr/local/include
# Default OpenCV libraries dir: /usr/local/lib

CPPFLAGS=-std=c++11 -I/usr/local/include
LDFLAGS=-Wall -Wextra
LDLIBS=-L/usr/local/lib \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_flann \
    -lopencv_features2d -lopencv_ml

all: bagofwords_classification

bagofwords_classification: bagofwords_classification.o
	 g++ $(LDFLAGS) -o bagofwords_classification bagofwords_classification.o $(LDLIBS)

bagofwords_classification.o: bagofwords_classification.cpp
	 g++ $(CPPFLAGS) -c bagofwords_classification.cpp
     
clean:
	 rm bagofwords_classification.o bagofwords_classification
