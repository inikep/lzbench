#CFLAGS = -O3
SRC =
INC = ../../inc_nx
TESTS = gzip_simple_test
#NXFLAGS = -DP9DBG

all:	$(TESTS)

gzip_vas.o:	../../lib/gzip_vas.c
	gcc $(CFLAGS) $(NXFLAGS) -I$(INC) -c ../../lib/gzip_vas.c

gzip_simple.o:	gzip_simple.c
	gcc $(CFLAGS) $(NXFLAGS) -I$(INC) -c gzip_simple.c

p9gzipsimple.a:	gzip_vas.o gzip_simple.o
	ar rcs p9gzipsimple.a gzip_vas.o gzip_simple.o
	ranlib p9gzipsimple.a

gzip_simple_driver.o:	gzip_simple_driver.c
	gcc $(CFLAGS) $(NXFLAGS) -I$(INC) -c gzip_simple_driver.c

gzip_simple_test:	p9gzipsimple.a gzip_simple_driver.o
	gcc $(CFLAGS) -o $@ gzip_simple_driver.o ./p9gzipsimple.a -lpthread

clean:
	rm -f $(TESTS) *.o *.c~ *.a *.h~ Makefile~
