PROG = glza
CC = gcc
CFLAGS = -O3 -DPRINTON -D_FILE_OFFSET_BITS=64
LDFLAGS = -lm -static -pthread

SRCS = GLZA.c GLZAcomp.c GLZAformat.c GLZAcompress.c GLZAencode.c GLZAdecode.c GLZAmodel.c
OBJS = $(SRCS:.c=.o)

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(PROG) $(OBJS)
