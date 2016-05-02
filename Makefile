NVCC := nvcc -arch sm_20
NVCC_FLAGS := -g -I/home/curtsinger/include -L/home/curtsinger/lib -ljansson
GCC := gcc
all: twitter

clean:
	@rm -f twitter

twitter: twitter.c util.o
	$(GCC) $(NVCC_FLAGS) -o twitter twitter.c util.o

util.o: util.c util.h
	gcc -c util.c


run: twitter
	LD_LIBRARY_PATH=/home/curtsinger/lib cat /home/curtsinger/data/tweets.json | ./twitter

