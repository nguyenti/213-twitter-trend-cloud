#include "util.c"

#define TREND_FETCH_TIME (5 * 60 * 1000) // should be 5 min
#define THREADS_PER_BLOCK 32


char* read_tweet(FILE * stream);
size_t read_trends(char ** trends, FILE * file);

//TODO: RENAME trend_maps, gpu_matrix, gpu_hashed_words
// COMPRESSEDLEN -> MAX_WORDS_PER_TWEET
// gpu_matrix -> gpu_word_counts_per_tweet DONE
// trend_maps -> gpu_word_counts_per_trend DONE

/* 
 * Populate a NUMTWEETS x word_count matrix with word counts per tweet
 *
 * @post gpu_word_counts_per_tweet[i * word_count + j] == # of occurrences 
 *                                                        of word j in tweet i
 *
 * gpu_tweets - the compressed representation of tweets
 * gpu_word_counts_per_tweet - the resulting matrix of word counts per tweet 
 * word_count - the number of distinct words in the current batch
 */
__global__ void compute_word_containment(int * gpu_tweets, 
                                         char * gpu_word_counts_per_tweet,
                                         int word_count);

/* 
 * Populate a NUMTRENDS x word_count matrix with word counts per trend
 *
 * @post gpu_word_counts_per_trend[i * word_count + j] == number of occurrences
 *                                     of word j in tweets that contain trend i
 *
 * gpu_word_counts_per_trend - a NUMTRENDS x word_count matrix
 * gpu_tweets_in_trends - array of number of tweets containing each trend 
 * gpu_trends - the compressed representation of trends
 * word_count - the number of distinct words in the batch
 * gpu_word_counts_per_tweet - the matrix of word counts per tweet 
 */
__global__ void get_trend_word_counts(int * gpu_word_counts_per_trend,
                                      int * gpu_tweets_in_trends,
                                      int * gpu_trends,
                                      int word_count,
                                      char * gpu_word_counts_per_tweet);

/*
 * Populate the matrix of words correlated with each trend


TODO: maybe combine with get_trend_word_counts? the resulting matrix is very
similar, has same dimensions

 * gpu_word_counts_per_trend - word counts for each trend
 * gpu_tweets_in_trends - number of tweets in a trend
 * total_word_counts - word counts for all words in all tweets
 * correlated_words - output, what's correlated
 * word_count - the number of distinct words in the batch
 */
__global__ void get_correlated_words(int * gpu_word_counts_per_trend,
                                     int * gpu_tweets_in_trends,
                                     int * total_word_counts,
                                     int * correlated_words,
                                     int word_count);


// Error-checking wrapper for cudaMalloc
void * CudaMalloc(size_t size, char * error_message) {
  void * ptr;
  if(cudaMalloc(&ptr, size) != cudaSuccess) {
  fprintf(stderr, "Failed to allocate %s on GPU\n", error_message);
    exit(2);
  }
  return ptr;
}


// Error-checking wrapper for cudaMemcpy
// error message should include what's being copied and a direction preposition
// e.g "tweets to" or "trends from"
void CudaMemcpy(void * destination, void * source, size_t size,
                enum cudaMemcpyKind direction, char * error_message) {
  if(cudaMemcpy(destination, source, size, direction) != cudaSuccess) {
    fprintf(stderr, "Failed to copy %s the GPU\n", error_message);
    exit(2);
  }
}

// Error-checking wrapper for cudaMemset to 0
void CudaZeroOut(void * ptr, size_t size, char * error_message) {
  if(cudaMemset(&ptr, 0, size) != cudaSuccess) {
    fprintf(stderr, "Failed to zero out %s on GPU\n", error_message);
    exit(2);
  }
}

// Error-checking wrapper for cudaDeviceSynchronize()
void CudaDeviceSynchronize(char * error_message) {
  if (cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "Failed to sync after %s\n", error_message);
    exit(2);
  }
}


// Main function
int main(int argc, char** argv) {
  // Timer for trend fetching. Should be every 5 minutes
  size_t start_time = time_ms() - TREND_FETCH_TIME - 1;

  // The trends
  char ** trends = (char **)malloc(sizeof(char *) * NUMTRENDS);
  // The tweets
  char tweets[NUMTWEETS][TWEETSIZE];

  // Array of compressed tweets and trends on host and device
  int compressed_tweets[NUMTWEETS][COMPRESSEDLEN];
  int compressed_trends[NUMTRENDS];
  int * gpu_tweets = (int*)CudaMalloc(sizeof(int) * NUMTWEETS * COMPRESSEDLEN,
                                "the tweets");
  int * gpu_trends = (int*)CudaMalloc(sizeof(int) * NUMTRENDS, "the trends");

    
  // Word counts per tweet and per trend, dynamically allocated later
  // We can also allocate them once, with maximum possible word count
  char * gpu_word_counts_per_tweet; 
  int * gpu_word_counts_per_trend; 

  // Number of tweets containing each trend
  int * gpu_tweets_in_trends = (int*)CudaMalloc(sizeof(int) * NUMTRENDS,
                                                    "tweets per trend matrix");
  // Word arrays
  char words[NUMTWEETS*COMPRESSEDLEN][TWEETSIZE];
  int total_word_counts[NUMTWEETS*COMPRESSEDLEN];
  int word_count = 0;

  // Correlated words
  int * gpu_correlated_words;
  int * correlated_words;
  
  // The pipe for the tweet stream
  int fd_tweets[2];
  int fd_trends[2];
  FILE * tweet_stream;

  char* tweet_args[] = {"cat", "tweets.json", NULL};
  char* trend_args[] = {"cat", "trends.json", NULL};

  // an error checking thing for forks
  int rc;
  
  int tweet_count = 0;

  int first_iteration = 1;
  size_t num_trends;
  
  // Open the tweet stream
  if (pipe (fd_tweets) < 0){
    perror("pipe error");
    exit(1);
  }
  rc = fork();
  if (rc < 0) { // error
    perror("Fork failed!");
    exit(1);
  } else if (rc == 0) { // child    
    pipe_stream(tweet_args, fd_tweets);
  } else {
    // parent: open the pipe as a file and keep it alive
    close(fd_tweets[1]);
    tweet_stream = fdopen(fd_tweets[0], "r");
  } 

  // Get the first tweet
  char* tweet = read_tweet(tweet_stream);
  
  // stop when out of tweets or trends or the user quits???
  while(tweet != NULL) {
    
    // TODO: Get stream of tweets and trends using forks and pipes
    if ((time_ms() - start_time) > TREND_FETCH_TIME) {
      start_time = time_ms();

      // open the pipe
      if (pipe (fd_trends) < 0){
        perror("pipe error");
        exit(1);
      }

      rc = fork();
      if (rc < 0) { // error
        perror("Fork failed!");
        exit(1);
      } else if (rc == 0) { // child
        pipe_stream(trend_args, fd_trends);
      } else { // parent
        close(fd_trends[1]);
        // read trends from stdout
        if (!first_iteration) {
          for (int i = 0; i < num_trends; i++) 
            free(trends[i]);
        } else { first_iteration = 0; }
        
        FILE * trend_stream = fdopen (fd_trends[0], "r");
        num_trends = read_trends(trends, trend_stream);
        fclose(trend_stream);
        if (num_trends < 1) {
          printf("Could not fetch trends\n");
          // TODO: cleanup
          exit(1);
        }
       
        
        // Copy trends onto the GPU
        if(cudaMemcpy(gpu_trends, compressed_trends, sizeof(int) * NUMTRENDS,
                      cudaMemcpyHostToDevice) != cudaSuccess) {
          fprintf(stderr, "Failed to copy trends to the GPU\n");
        }
      }
    }
    
    // save tweet by copying
    strncpy(tweets[tweet_count], tweet, TWEETSIZE);

    // Clean and compress the tweet
    clean_string(tweet);
    compress_str(tweet,
                 compressed_tweets[tweet_count],
                 words,
                 total_word_counts,
                 &word_count);
    free(tweet);
    

    if (tweet_count >= NUMTWEETS - 1) {
      
      // TESTING
      for (int i = 0; i < num_trends; i++)
         printf("trend #%d: %s\n", i, trends[i]);
      for (int i = 0; i < tweet_count; i++) 
        printf("tweet #%d: %s\n", i, tweets[i]);
      for (int i = 0; i < word_count; i++) 
        printf("word #%d: %s\n", i, words[i]);

      
      // Copy compressed tweets onto the GPU
      CudaMemcpy(gpu_tweets, compressed_tweets, sizeof(int) * NUMTWEETS *
                 COMPRESSEDLEN, cudaMemcpyHostToDevice, "tweets to");

      // Figure out trends' compressed values (indices in the word array)
      for (int i = 0; i < num_trends; i++) {
        // go through words array to find each trend's index
        // TODO sort and use binary search
        for (int j = 0; j < word_count; j++) {
          if (strcmp(trends[i], words[j]) == 0) {
            compressed_trends[i] = j;
            break;
          }
        }
      }

      //TESTING
       for (int i = 0; i < num_trends; i++)
         printf("Compressed trend %s - %d\n", trends[i], compressed_trends[i]);
       
      // Copy compressed trends onto the GPU
      CudaMemcpy(gpu_trends, compressed_trends, sizeof(int) * NUMTRENDS,
                 cudaMemcpyHostToDevice, "trends to");
      
      // Allocate and zero wc per tweet matrix
      char * gpu_word_counts_per_tweet =(char*)CudaMalloc(sizeof(char)*NUMTWEETS
                                                         * word_count,
                                                         "wc per tweet matrix");
      CudaZeroOut(gpu_word_counts_per_tweet, sizeof(char) * NUMTWEETS
                  * word_count, "word counts per tweet matrix");
      
      // Allocate wc per trend matrix
      //TODO do we need to zero this? no
      int * gpu_word_counts_per_trend = (int*)CudaMalloc(sizeof(int)*NUMTRENDS
                                                         * word_count,
                                                         "wc per trend matrix");
      // Allocate and copy total_word_counts onto GPU
      int * gpu_total_word_counts = (int*)CudaMalloc(sizeof(int) * word_count,
                                                     "total word counts");
      CudaMemcpy(gpu_total_word_counts, total_word_counts,sizeof(int) *
                 word_count, cudaMemcpyHostToDevice, "total wcs to");

      // Zero out gpu_tweets_in_trends on GPU 
      CudaZeroOut(gpu_tweets_in_trends, sizeof(int) * NUMTRENDS,
                  "word counts per trend matrix");

      // Allocate enough correlated words for each trend
      gpu_correlated_words = (int*)CudaMalloc(sizeof(int) * NUMTRENDS *
                                              word_count/2, "correlated_words");
      
      // Make tweets x words matrix of counts  
      compute_word_containment<<<1, NUMTWEETS>>>(gpu_tweets,
                                                 gpu_word_counts_per_tweet,
                                                 word_count);
      // Wait for the kernel to finish
      CudaDeviceSynchronize("computing word counts per tweet");
      
      // To make gpu_word_counts_per_trend, add rows of the tweet X word matrix
      // that correpond to each trend
      get_trend_word_counts<<<1, NUMTRENDS>>>(gpu_word_counts_per_trend,
                                              gpu_tweets_in_trends,
                                              gpu_trends,
                                              word_count,
                                              gpu_word_counts_per_tweet);

      // Wait for the kernel to finish
      CudaDeviceSynchronize("computing word counts per trend");
      
      // Find word sets correlated with each topic 
      get_correlated_words<<<1, NUMTRENDS>>>(gpu_word_counts_per_trend,
                                             gpu_tweets_in_trends,
                                             total_word_counts,
                                             gpu_correlated_words,
                                             word_count);

      
      // Wait for the kernel to finish
      CudaDeviceSynchronize("computing correlated words");

      // Allocate and copy correlated words onto CPU
      correlated_words = (int*)malloc(sizeof(int) * NUMTRENDS * word_count/2);
      CudaMemcpy(correlated_words, gpu_correlated_words, sizeof(int) *
                 NUMTRENDS * word_count/2, cudaMemcpyDeviceToHost,
                 "correlated words from");

      //TESTING
      for (int i = 0; i < NUMTRENDS; i++) {
        printf("Words correlated with trend %s:\n", trends[i]);
        for (int j = 0; j < word_count ; j++) {
          int word_index = correlated_words[word_count/2 * i + j];
          if (word_index == END_OF_TWEET) break;
          printf("%s(%d), ", words[word_index], word_index);
        }
      }
      
      // TODO: Create word clouds with external tools (go to 8 if it doesn’t
      //   work out)
      // TODO: If time allows: Implement weighing words by importance (tf-idf)
      // TODO: If time allows: Explore other uses of the same output data: graph
      //   building, clustering, or term evolution over time
      // TODO: If time allows: Rewrite compress_str to use insertion sort and binary search
      //   maybe with an auxiliary data structure

      
      // Free stuff: gpu_word_counts_per_trend
      //             gpu_word_counts_per_tweet
      //             gpu_total_word_counts
      
      
      word_count = 0;
      tweet_count = 0;
      // TESTING
      return 0;
    } // for each NUMTWEETS tweets
    
    // Read the next tweet  
    tweet = read_tweet(tweet_stream);
    tweet_count++;
  }
  // Close the pipe and the file
  fclose(tweet_stream);
  close(fd_tweets[0]);

  // Free CUDA stuff
  
  //for (int i = 0; i < NUMTWEETS; i++)
  //  free(tweets[i]);
  //free(tweets);
  
  for (int i = 0; i < NUMTRENDS; i++)
    free(trends[i]);
  free(trends);
  
  return 0;
}
  

  
  
/* Get inputs from the feed */

// Returns the size of the trends array
// Make sure to allocate the trends array!
// char ** trends = (char **)malloc(sizeof(char *) * NUMTRENDS);
size_t read_trends(char ** trends, FILE * file) {

  // for using getline
  static char* line = NULL;
  static size_t line_maxlen = 0;
  
  // Loop until we read one valid json array or reach the end of the input
  while(getline(&line, &line_maxlen, file) > 0) {
    // Parse the JSON body
    json_error_t error;
    // The outer array, hypothetically
    json_t* root = json_loads(line, 0, &error);
  
    // Skip over lines with errors
    if(!root) continue;
  
    // Skip over lines that aren't JSON nonempty arrays
    if(!json_is_array(root) || json_array_size(root) < 1) {
      json_decref(root);
      continue;
    }

    // Get the first object in the array
    json_t * first_object = json_array_get(root, 0);

    // Get the json trends array
    json_t * json_trends_array = json_object_get(first_object, "trends");

    size_t arr_size;
     // Make sure 'trends' is a nonempty array
    if(!json_is_array(json_trends_array) ||
       (arr_size = json_array_size(json_trends_array)) < 1) {
      json_decref(json_trends_array);
      json_decref(first_object);
      json_decref(root);
      continue;
    }

    size_t i;
    // Read every trend into a regular C array
    for (i = 0; i < arr_size &&  i < NUMTRENDS; i++) {
      json_t * json_trend_obj = json_array_get(json_trends_array, i);
      
      // Get the name of the trend
      json_t* text = json_object_get(json_trend_obj, "name");
  
      // If there was no text, skip this t
      if(!json_is_string(text)) {
        json_decref(root);
        continue;
      }
  
      // Get the string out of the JSON text value
      const char* json_text = json_string_value(text);
  
      // Got a trend! Copy just the trend text to an allocated buffer
      trends[i] = (char*)malloc(sizeof(char) * (strlen(json_text) + 1));
      strcpy(trends[i], json_text);

      // Only use the first word of the trend
      trends[i] = strtok(trends[i], " ");
      
      // Release this reference to the JSON object
      json_decref(text);
      json_decref(json_trend_obj);
    }

    // Release references to JSON objects
    json_decref(json_trends_array);
    json_decref(first_object);
    json_decref(root);

    //free(line);
    
    // Return the number of trends read
    return i;
  }
  
  // Ran out of input. Just return 0
  return 0;
} // read_trends


char* read_tweet(FILE * stream) {
  static char* line = NULL;
  static size_t line_maxlen = 0;
  ssize_t line_length;
  
  // Loop until we read one valid tweet or reach the end of the input
  while((line_length = getline(&line, &line_maxlen, stream)) > 0) {
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

    //free(line);
    
    // Return the result
    return tweet_text;
  }
  
  // Ran out of input. Just return NULL
  return NULL;
}


// N tweets x K trends
// gpu_word_counts_per_tweet must be zeroed out
__global__ void compute_word_containment(int * gpu_tweets,
                                         char * gpu_word_counts_per_tweet,
                                         int word_count) {
  int index =  threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
  if (index < NUMTWEETS) {
    for (int i = 0; i < COMPRESSEDLEN
           && gpu_tweets[COMPRESSEDLEN * index + i] != END_OF_TWEET; i++) {
      // each cell of gpu_tweets *is* the index of the word
      gpu_word_counts_per_tweet[word_count * index +
                                gpu_tweets[COMPRESSEDLEN * index + i]]++;
    }
  }
}


// TODO: make it go across rows (one thread per tweet), not for every trend
__global__ void get_trend_word_counts(int * gpu_word_counts_per_trend,
                                      int * gpu_tweets_in_trends,
                                      int * gpu_trends,
                                      int word_count,
                                      char * gpu_word_counts_per_tweet) {
  int trend_index =  threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
  if (trend_index < NUMTRENDS) {
    int trend_map_index = trend_index * word_count;
    int trend_word_index = gpu_trends[trend_index];
    for (int i = 0; i < NUMTWEETS; i++) { // for every tweet
      if (gpu_word_counts_per_tweet[i * word_count + trend_word_index] > 0) {
        // if the trend is present in the tweet
        for (int j = 0; j < word_count; j++) {
          // get the word counts for all the words in the tweet
          gpu_word_counts_per_trend[trend_map_index + j] +=
            gpu_word_counts_per_tweet[i * word_count + j];
        }
        gpu_tweets_in_trends[trend_index]++;
      }
    }
  }
}

// Make it go across rows, not columns (with word_count threads)
__global__ void get_correlated_words(int * gpu_word_counts_per_trend,
                                     int * gpu_tweets_in_trends,
                                     int * total_word_counts,
                                     int * correlated_words,
                                     int word_count) {
  int trend_index =  threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
  if (trend_index < NUMTRENDS) {
    int num_correlated_words = 0;
    for (int i = 0; i < word_count; i++) {
      if (gpu_word_counts_per_trend[word_count * trend_index + i] /
          (double) gpu_tweets_in_trends[trend_index] >
          CORRELATION_FACTOR * total_word_counts[i] / (double) word_count) {
        // got a correlated word! Record it
        correlated_words[word_count * trend_index + num_correlated_words++] = i;
      }
      // Signify the end of correlated words
      correlated_words[word_count * trend_index + num_correlated_words] =
        END_OF_TWEET;
    }
  }
}
