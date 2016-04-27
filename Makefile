NVCC := nvcc -arch sm_20
NVCC_FLAGS := -g -I/home/curtsinger/include -L/home/curtsinger/lib -ljansson

all: twitter

clean:
	@rm -f twitter

twitter: twitter.cu util.o
	$(NVCC) $(NVCC_FLAGS) -o twitter twitter.cu util.o

util.o: util.c util.h
	gcc -c util.c


run: twitter
	LD_LIBRARY_PATH=/home/curtsinger/lib cat /home/curtsinger/data/tweets.json | ./twitter

