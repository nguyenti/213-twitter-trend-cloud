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

#define NUMTRENDS 50 // Never should be more than 50, per the API
#define NUMTWEETS 1000
#define COMPRESSEDLEN 36
#define TWEETSIZE 141

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

//Citation: https://en.wikipedia.org/wiki/Jenkins_hash_function
uint32_t hash_func (char *key)
{
  int len = strlen(key);
  uint32_t hash, i;
  for(hash = i = 0; i < len; ++i)
    {
      hash += key[i];
      hash += (hash << 10);
      hash ^= (hash >> 6);
    }
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  // avoid returning zeros
  return hash == 0 ? 1 : hash;
}

// Takes out the punctuation from tweets
void clean_string(char* string) {
  char ch;
  int i = 0;
  while((ch = string[i]) != '\0') {
    if (ispunct(ch) && ch != '#') {
      // replace punctuation with whitespace
      string[i] = ' ';
    }
    tolower(string[i]); // downcases the characters 
    i++;
  }
}

/* compress a string into an array of hashed numbers.
 * Add each word to our word_count counter
 * @para: string is already cleaned and compressed is a pointer to 
 *        an array of 32 int  
 */
void compress_str (char* string, int* compressed, char words[][TWEETSIZE],
                   int * hash, int * total_word_counts, int * word_count) {
  int index = 0;
  char* word[TWEETSIZE];
  while ((word[index] = strtok(string, " \0\n\t")) != NULL &&
         index < COMPRESSEDLEN){
    string = NULL;
    if (strlen(word[index]) >= 3) { // if the word is appropriate size
      
      // Check if the word is already in our map
      int i;
      for (i = 0; i < *word_count; i++) {
        if (compressed[index] == hash[i]) {
          // it is! So we increment the count
          total_word_counts[i]++;
          break;
        }
      }
      // If it wasn't found, add it to our map
      if (i == *word_count) {
        strncpy(words[*word_count], word[index], TWEETSIZE);
        hash[*word_count] = compressed[index];
        total_word_counts[*word_count]++;
        (*word_count)++;
      }
      // Use the index of the word for compressed representation
      compressed[index] = i; //hash_func(word[index]);
    }
    index++;
  }
  compressed[index] = 0;
}



#endif
