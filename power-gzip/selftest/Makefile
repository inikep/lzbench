CC ?= gcc
CFLAGS ?= -O3
INC = ./inc
SRC = gzfht_test.c gunz_test.c
OBJ = $(SRC:.c=.o)
TESTS = gzfht_test gunz_test
EXTRA_SOURCES = gzip_vas.c

all:	$(TESTS)

$(OBJ): %.o: %.c
	$(CC) $(CFLAGS) -I$(INC) -c $<

$(TESTS): $(OBJ)
	$(CC) $(CFLAGS) -I$(INC) -o $@ $@.o $(EXTRA_SOURCES)

run_tests: $(TESTS)
	./gzfht_test gzip_vas.c
	./gunz_test gzip_vas.c.nx.gz

clean:
	rm -f $(TESTS) *.o *~ *.gz *.gunzip
