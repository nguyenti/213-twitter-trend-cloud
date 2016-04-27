#include <string.h>
#include <jansson.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <errno.h>

#include "util.h"
#define TREND_FETCH_TIME (5 * 60 * 1000)

// Read the tweet in from stdin
char* read_tweet();


// Main function
int main(int argc, char** argv) {
  char* tweet = read_tweet();

  // Timer for trend fetching. Should be every 5 minutes
  size_t start_time = time_ms() - TREND_FETCH_TIME;
  
  while(tweet != NULL) {
    printf("%s\n", tweet);
    
    // TODO: Get stream of tweets and trends using forks and pipes
    if ((time_ms() - start_time) > TREND_FETCH_TIME) {
      start_time = time_ms();
      int fd[2];    // provide file descriptor pointer array for pipe 
      /* within pipe:
         fd[0] will be input end
         fd[1] will be output end */


      if (pipe (fd) < 0){
        perror("pipe error");
        exit (2);
      }

      int rc = fork();
      if (rc < 0) { // error
        perror("Fork failed!");
        exit(2);
      } else if (rc == 0) { // child
        
      } else { // parent

      }
    }
    
    
    // TODO: Clean data
    // TODO: Make an NxN intersection matrix
    // TODO: Make an NxK topic containment bit matrix
    // TODO: Find word sets correlated with each topic and compute correlation
    //   coefficients
    // TODO: Create word clouds with external tools (go to 8 if it doesn’t
    //   work out)
    // TODO: If time allows: Implement weighing words by importance (tf-idf)
    // TODO: If time allows: Explore other uses of the same output data: graph
    //   building, clustering, or term evolution over time
    
    // Free the tweet string
    free(tweet);
    
    // Read the next tweet
    tweet = read_tweet();
  }
  
  return 0;
}


char* read_tweet() {
  static char* line = NULL;
  static size_t line_maxlen = 0;
  ssize_t line_length;
  
  // Loop until we read one valid tweet or reach the end of the input
  while((line_length = getline(&line, &line_maxlen, stdin)) > 0) {
    // Parse the JSON body
    json_error_t error;
    json_t* root = json_loads(line, 0, &error);
  
    // Skip over lines with errors
    if(!root) continue;
  
    // Skip over lines that aren't JSON objects
    if(!json_is_object(root)) {
      json_decref(root);
      continue;
    }
  
    // Get the text of the tweet
    json_t* text = json_object_get(root, "text");
  
    // If there was no text, skip this tweet
    if(!json_is_string(text)) {
      json_decref(root);
      continue;
    }
  
    // Get the string out of the JSON text value
    const char* json_text = json_string_value(text);
  
    // Got a tweet! Copy just the tweet text to an allocated buffer
    char* tweet_text = (char*)malloc(sizeof(char) * (line_length + 1));
    strcpy(tweet_text, json_text);
    
    // Release this reference to the JSON object
    json_decref(root);
    
    // Return the result
    return tweet_text;
  }
  
  // Ran out of input. Just return NULL
  return NULL;
}
