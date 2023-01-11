CFLAGS = -g -O3 -flto -fwhole-program -Wall -Wextra
ppmd-mini: ppmd-mini.c lib/Ppmd*.c
	$(CC) -Ilib $(CFLAGS) -o $@ $^
