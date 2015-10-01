CC=gcc
CFLAGS=-Wall -O3

all: main

%: %.c
	$(CC) $(CFLAGS) -pthread $< -lm

clean:
	rm a.out
