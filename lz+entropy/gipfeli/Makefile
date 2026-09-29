CC=g++
SRC=decompress.cc entropy.cc entropy_code_builder.cc lz77.cc gipfeli-internal.cc gipfeli_test.cc
OBJ=$(SRC:.cc=.o)
CFLAGS=-Wall -Werror -Wno-sign-compare -Wno-strict-aliasing -O3

all: gipfeli_test

clean:
	rm -f *.o gipfeli_test

%.o: %.cc
	$(CC) -c -o $@ $< $(CFLAGS)
gipfeli_test:  $(OBJ)
	$(CC) -o $@ $(OBJ) $(CFLAGS)
