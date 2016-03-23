
all: makexr.cpp
	g++ -O3 --std=c++11 -Wall -I. -I/usr/include/OpenEXR makexr.cpp -o makexr.bin -lpng -lz -lIlmImf -lIex
	
clean:
	rm makexr.bin
