PROG = lab3
CFLAGS = -g -fPIC -m64 -Wall -std=c++11
LFLAGS= -fPIC -m64 -Wall
CC = mpiCC

all: $(PROG)

lab3.o: lab3.cpp
	$(CC) $(CFLAGS) -c lab3.cpp

lab3 : lab3.o
	$(CC) $(LFLAGS) lab3.o -o lab3

run:
	mpirun --hostfile hostfile -np 64 lab3

ps:
	ps -fu $$USER

clean:
	/bin/rm -f *~
	/bin/rm -f *.o
