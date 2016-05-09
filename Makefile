NVCC := nvcc -arch sm_20
NVCC_FLAGS := -g -I/home/curtsinger/include -L/home/curtsinger/lib -ljansson
GCC := gcc
all: twitter

clean:
	@rm -f twitter

twitter: twitter.cu util.c
	$(NVCC) $(NVCC_FLAGS) -o twitter twitter.cu util.c

run: twitter
	LD_LIBRARY_PATH=/home/curtsinger/lib ./twitter $(TWEET_OAUTH) $(TREND_OAUTH)

