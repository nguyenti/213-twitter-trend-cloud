#ifndef __UTIL_C__
#define __UTIL_C__

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

#define NUMTRENDS 40 // Never should be more than 50, per the API
#define NUMTWEETS 1000 // 1000
#define COMPRESSEDLEN 36
#define TWEETSIZE 141
// a number that cannot possibly represent a compressed word because it exceeds
// the number of possible words can signify the end of a tweet
#define END_OF_TWEET (-1)
#define CORRELATION_FACTOR 1

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

// Takes out the punctuation from tweets
void clean_string(char* string) {
  char ch;
  int i = 0;
  while((ch = string[i]) != '\0') {
    if (!isalpha(ch) && ch != '#' && ch != '@') {
      // replace punctuation with whitespace
      string[i] = ' ';
    }
    // downcase the characters 
    string[i] = tolower(string[i]); 
    i++;
  }
}

/* Compress a string into an array of ints.
 * Add each word to our word_count counter
 *
 * @para: string is already cleaned
 *        compressed is a pointer to an allocated array of 32 int
 *        words is an allocated array of all words encountered so far
 *        total_word_counts -- 1D array of counts for every word in words
 *        word_count -- number of distinct words encountered so far       
 */
void compress_str (char* string, int* compressed, char words[][TWEETSIZE],
                   int * total_word_counts, int * word_count) {
  int index = 0;
  char* tweet[TWEETSIZE];
  while ((tweet[index] = strtok(string, " \0\n\t")) != NULL &&
         index < COMPRESSEDLEN){
    string = NULL;
    // if the word is appropriate size and not a twitter handle
    if (strlen(tweet[index]) >= 3 && tweet[index][0] != '@') {
      // Check if the word is already in our map
      int i;
      for (i = 0; i < *word_count; i++) {
        if (strcmp(tweet[index], words[i]) == 0) {
          // it is! So we increment the count
          total_word_counts[i]++;
          break;
        }
      }
      // If it wasn't found, add it to our map
      if (i == *word_count) {
        strncpy(words[*word_count], tweet[index], TWEETSIZE);
        total_word_counts[*word_count] = 1;
        (*word_count)++;
      }
      // Use the index of the word for compressed representation
      compressed[index] = i;
      index++;
    }
  }
  // Indicate the end of the compressed representation
  compressed[index] = END_OF_TWEET;
}



#endif
