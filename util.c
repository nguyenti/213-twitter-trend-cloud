#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <jansson.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <errno.h>

#define NUMTRENDS 50 // Never should be more than 50, per the API
#define NUMTWEETS 1000

/**
 * Pipe a stream from a child process
 * Must be called from the child after the fork
 */
void pipe_stream(char ** command, int * pipe_fd) {  
  // close input, leave output open
  close (pipe_fd[0]);
  // set stdout to pipe_fd
  if (pipe_fd[1] !=  STDOUT_FILENO) {
    if (dup2(pipe_fd[1], STDOUT_FILENO) != STDOUT_FILENO){
      perror("dup2 error for standard output");
      exit(1);
    }
    close(pipe_fd[1]); 
  }
  // now stdout goes to the pipe
  execvp(command[0], command);
  perror("execvp failed");
}// pipe_stream

/**
 * Get the time in milliseconds since UNIX epoch
 */
size_t time_ms() {
  struct timeval tv;
  if(gettimeofday(&tv, NULL) == -1) {
    perror("gettimeofday");
    exit(2);
  }
  
  // Convert timeval values to milliseconds
  return tv.tv_sec*1000 + tv.tv_usec/1000;
}

