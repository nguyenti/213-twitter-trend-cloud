#include "util.c"

#define TREND_FETCH_TIME (5 * 60 * 1000) // should be 5 min
#define THREADS_PER_BLOCK 32

 #define NUMCURLARGS 9
#define NUMCATARGS 3

/*
 * Read a tweet from the stream 
 */
char* read_tweet(FILE * stream);

/*
 * Read the trends from the stream 
 */
size_t read_trends(char trends[][TWEETSIZE], FILE * file);

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
 * num_trends - number of trend being considered 
 */
__global__ void get_trend_word_counts(int * gpu_word_counts_per_trend,
                                      int * gpu_tweets_in_trends,
                                      int * gpu_trends,
                                      int word_count,
                                      char * gpu_word_counts_per_tweet,
                                      int num_trends);

/*
 * Populate the matrix of words correlated with each trend
 * gpu_word_counts_per_trend - word counts for each trend
 * gpu_tweets_in_trends - number of tweets in a trend
 * total_word_counts - word counts for all words in all tweets
 * correlated_words - output, what's correlated
 * weights - frequency of each of the correlated words
 * word_count - the number of distinct words in the batch
 * num_trends - number of trend being considered 
 */
__global__ void get_correlated_words(int * gpu_word_counts_per_trend,
                                     int * gpu_tweets_in_trends,
                                     int * total_word_counts,
                                     int * correlated_words,
                                     int * weights,
                                     int word_count,
                                     int num_trends);


// Error-checking wrapper for cudaMalloc
void * CudaMalloc(size_t size, char * error_message);

/*
 * Error-checking wrapper for cudaMemcpy
 * error message should include what's being copied and a direction preposition
 * e.g. "tweets to" or "trends from" 
 */
void CudaMemcpy(void * destination, void * source, size_t size,
                enum cudaMemcpyKind direction, char * error_message);

// Error-checking wrapper for cudaMemset to 0
void CudaZeroOut(void * ptr, size_t size, char * error_message);

// Error-checking wrapper for cudaDeviceSynchronize()
void CudaDeviceSynchronize(char * error_message);


// Main function
int main(int argc, char** argv) {

  // Timer for trend fetching. Should be every 5 minutes
  size_t start_time = time_ms() - TREND_FETCH_TIME - 1;

  // The trends
  char trends[NUMTRENDS][TWEETSIZE];
  // The tweets
  char tweets[NUMTWEETS][TWEETSIZE];

  // Array of compressed tweets and trends on host and device
  int compressed_tweets[NUMTWEETS][COMPRESSEDLEN];
  int compressed_trends[NUMTRENDS];
  int * gpu_tweets = (int*)CudaMalloc(sizeof(int) * NUMTWEETS * COMPRESSEDLEN,
                                "the tweets");
  int * gpu_trends = (int*)CudaMalloc(sizeof(int) * NUMTRENDS, "the trends");

  // Number of tweets containing each trend
  int * gpu_tweets_in_trends = (int*)CudaMalloc(sizeof(int) * NUMTRENDS,
                                                    "tweets per trend matrix");
  // Word arrays
  char words[NUMTWEETS*COMPRESSEDLEN][TWEETSIZE];
  int total_word_counts[NUMTWEETS*COMPRESSEDLEN];
  int word_count = 0;

  // Correlated words and their weights
  int * gpu_correlated_words;
  int * correlated_words;
  int * gpu_weights;
  int * weights;
  
  // The pipe for the tweet stream
  int fd_tweets[2];
  
  FILE * tweet_stream;
  char ** tweet_args;
  char ** trend_args;

  // Hardcoded arguments for the curl and cat commands
  char * curl_tweet_args[] = {"curl", "--get",
                         "https://stream.twitter.com/1.1/statuses/sample.json",
                              "--data", "language=en",
                              "--header", NULL, "--verbose", NULL};
  char * curl_trend_args[] =  {"curl", "--get",
                               "https://api.twitter.com/1.1/trends/place.json",
                               "--data", "id=1", "--header", NULL, "--verbose",
                               NULL};
  char * cat_tweet_args[] = {"cat", "new_tweets.json", NULL};
  char* cat_trend_args[] = {"cat", "new_trends.json", NULL};

  // If we have enough command-line arguments, use curl
  if (argc == 3) {
    tweet_args = (char **) malloc(sizeof(char*) * NUMCURLARGS);
    trend_args = (char **) malloc(sizeof(char*) * NUMCURLARGS);
    for (int i = 0; i < NUMCURLARGS; i++) {
      if (i == 6) {
        tweet_args[i] = argv[1];
        trend_args[i] = argv[2];
      } else {
        tweet_args[i] = curl_tweet_args[i];
        trend_args[i] = curl_trend_args[i];
      }
    }
  } else { // otherwise, use cat
    tweet_args = (char **) malloc(sizeof(char*) * NUMCATARGS);
    trend_args = (char **) malloc(sizeof(char*) * NUMCATARGS);
    for (int i = 0; i < NUMCATARGS; i++) {
      tweet_args[i] = cat_tweet_args[i];
      trend_args[i] = cat_trend_args[i];
    }
  }
  
  // File for persisting cloud data
  FILE * cloud_output = fopen("output/clouds.txt", "a+");
  
  // an error checking variable for forks
  int rc;
  
  int tweet_count = 0;
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
    if (tweet_stream == NULL) {
      fprintf(stderr, "Failed to open the tweet pipe\n");
      exit(1);
    }
  } 

  // Get the first tweet
  char* tweet = read_tweet(tweet_stream);
  
  // Don't stop unless ran out of tweets
  while(tweet != NULL) {
    // Check if it is time to fetch new trends
    if ((time_ms() - start_time) > TREND_FETCH_TIME) {
      
      start_time = time_ms();
      num_trends = 0;
      // open a pipe
      int fd_trends[2];
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
        
        FILE * trend_stream = fdopen (fd_trends[0], "r");
        if (tweet_stream == NULL) {
          fprintf(stderr, "Failed to open the trend pipe\n");
          exit(1);
        }
        num_trends = read_trends(trends, trend_stream);
        fclose(trend_stream);
        if (num_trends < 1) {
          fprintf(stderr, "Could not fetch trends\n");
          break;
        }
      }  
    } // optionally fetching new trends
        
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
        
    // Make wird clouds for every NUMTWEETS-sized batch of tweets
    if (tweet_count >= NUMTWEETS - 1) {
      size_t start = time_ms();   
      // Copy compressed tweets onto the GPU
      CudaMemcpy(gpu_tweets, compressed_tweets, sizeof(int) * NUMTWEETS *
                 COMPRESSEDLEN, cudaMemcpyHostToDevice, "tweets to");

      // Figure out trends' compressed values (indices in the word array)
      for (int i = 0; i < num_trends; i++) {
        // go through words array to find each trend's index
        // TODO sort and use binary search
        for (int j = 0; j < word_count; j++) {
          if (strncmp(trends[i], words[j], TWEETSIZE) == 0) {
            compressed_trends[i] = j;
            break;
          }
          else compressed_trends[i] = END_OF_TWEET; 
        }
      }

      // Copy compressed trends onto the GPU
      CudaMemcpy(gpu_trends, compressed_trends, sizeof(int) * NUMTRENDS,
                 cudaMemcpyHostToDevice, "trends to");
      
      // Allocate and zero wc per tweet matrix
      char * gpu_word_counts_per_tweet =(char*)CudaMalloc(sizeof(char)*NUMTWEETS
                                                         * word_count,
                                                         "wc per tweet matrix");
      CudaZeroOut(gpu_word_counts_per_tweet, sizeof(char) * NUMTWEETS
                  * word_count, "word counts per tweet matrix");
      
      // Allocate and zero wc per trend matrix
      int * gpu_word_counts_per_trend = (int*)CudaMalloc(sizeof(int)*NUMTRENDS
                                                         * word_count,
                                                         "wc per trend matrix");
      CudaZeroOut(gpu_word_counts_per_trend, sizeof(int) * NUMTRENDS
                  * word_count, "word counts per trend matrix");
      
      // Allocate and copy total_word_counts onto GPU
      int * gpu_total_word_counts = (int*)CudaMalloc(sizeof(int) * word_count,
                                                     "total word counts");
      CudaMemcpy(gpu_total_word_counts, total_word_counts, sizeof(int) *
                 word_count, cudaMemcpyHostToDevice, "total wcs to");

      // Zero out gpu_tweets_in_trends on GPU 
      CudaZeroOut(gpu_tweets_in_trends, sizeof(int) * NUMTRENDS,
                  "word counts per trend matrix");

      // Allocate enough correlated words for each trend
      gpu_correlated_words = (int*)CudaMalloc(sizeof(int) * NUMTRENDS *
                                              word_count, "correlated words");

      gpu_weights = (int*)CudaMalloc(sizeof(int) * NUMTRENDS *
                                     word_count, "correlated words' weights");
      
      // Make tweets*words matrix of counts (populate gpu_word_counts_per_tweet)
      compute_word_containment<<<1, NUMTWEETS>>>(gpu_tweets,
                                                 gpu_word_counts_per_tweet,
                                                 word_count);
      // Wait for the kernel to finish
      CudaDeviceSynchronize("computing word counts per tweet");

      // Make gpu_word_counts_per_trend by adding rows of the tweet*word matrix
      // that correpond to each trend
      get_trend_word_counts<<<1, num_trends>>>(gpu_word_counts_per_trend,
                                               gpu_tweets_in_trends,
                                               gpu_trends,
                                               word_count,
                                               gpu_word_counts_per_tweet,
                                               num_trends);

      // Wait for the kernel to finish
      CudaDeviceSynchronize("computing word counts per trend");

      // Find word sets correlated with each topic 
      get_correlated_words<<<1, num_trends>>>(gpu_word_counts_per_trend,
                                              gpu_tweets_in_trends,
                                              gpu_total_word_counts,
                                              gpu_correlated_words,
                                              gpu_weights,
                                              word_count,
                                              num_trends);

      
      // Wait for the kernel to finish
      CudaDeviceSynchronize("computing correlated words");

      // Allocate and copy correlated words onto CPU
      correlated_words = (int*)Malloc(sizeof(int) * NUMTRENDS * word_count,
                                      "correlated words");
      CudaMemcpy(correlated_words, gpu_correlated_words, sizeof(int) *
                 NUMTRENDS * word_count, cudaMemcpyDeviceToHost,
                 "correlated words from");

      // Allocate and copy weights of the correlated words
      weights = (int*)Malloc(sizeof(int) * NUMTRENDS * word_count, "weights");
      CudaMemcpy(weights, gpu_weights, sizeof(int) *
                 NUMTRENDS * word_count, cudaMemcpyDeviceToHost,
                 "weights from");

      // Write to file
      for (int i = 0; i < num_trends; i++) {
        fprintf(cloud_output, "%s\n", trends[i]);
        for (int j = 0; j < word_count ; j++) {
          int word_index = correlated_words[word_count * i + j];
          if (word_index == END_OF_TWEET) break;
          fprintf(cloud_output, "%s:%d,", words[word_index],
                 weights[word_count * i + j]);
        }
        fprintf(cloud_output, "\n");
      }
      
      // Free dynamically allocated arrays 
      cudaFree(gpu_word_counts_per_trend);
      cudaFree(gpu_word_counts_per_tweet);
      cudaFree(gpu_total_word_counts);
      cudaFree(gpu_correlated_words);
      cudaFree(gpu_weights);

      free(correlated_words);
      free(weights);

      // Zero out total_word_counts
      for (int i = 0; i < word_count; i++)
        total_word_counts[i] = 0;
      
      // Zero counters
      word_count = 0;
      tweet_count = 0;

      // For timing purposes
      //size_t end = time_ms();
      //printf("%d,%zu\n", NUMTWEETS, end-start);
    } // for each NUMTWEETS tweets
   
    // Read the next tweet  
    tweet = read_tweet(tweet_stream);
    tweet_count++;
  }// while(tweet != NULL)

  // Clean up after running out of trends or tweets
  
  // Close the pipe and the file
  fclose(tweet_stream);
  close(fd_tweets[0]);
  fclose(cloud_output);
  
  // Free CUDA stuff
  cudaFree(gpu_tweets);
  cudaFree(gpu_trends);
  cudaFree(gpu_tweets_in_trends);
  
  return 0;
}
  
  
/****************************
 * Get inputs from the feed *
 ****************************/

// Returns the size of the trends array
// trends should be allocated
size_t read_trends(char trends[NUMTRENDS][TWEETSIZE], FILE * file) {
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
    if(!root){
      fprintf(stderr, "Invalid json line encountered while reading trends\n");
      continue;
    }
  
    // Skip over lines that aren't JSON nonempty arrays
    if(!json_is_array(root) || json_array_size(root) < 1) {
      fprintf(stderr, "Invalid or empty outer json array encountered while reading trends\n");
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
      fprintf(stderr, "Invalid or empty inner json array encountered while reading trends\n");
      json_decref(json_trends_array);
      json_decref(first_object);
      json_decref(root);
      continue;
    }

    size_t i;
    size_t trend_index = 0;
    // Read every trend into a regular C array
    for (i = 0; i < arr_size &&  i < NUMTRENDS
           && trend_index < NUMTRENDS; i++) {
      json_t * json_trend_obj = json_array_get(json_trends_array, i);
      
      // Get the name of the trend
      json_t* text = json_object_get(json_trend_obj, "name");
  
      // If there was no text, skip this 
      if(!json_is_string(text)) {
        fprintf(stderr,
                "Invalid json string encountered while reading trends\n");
        json_decref(root);
        continue;
      }
  
      // Get the string out of the JSON text value
      const char* json_text = json_string_value(text);

      char * temp = (char *) malloc(sizeof(char) * TWEETSIZE);
      // Got a trend! Copy just the trend text 
      strncpy(temp, json_text, TWEETSIZE);

      // Clean the trend
      clean_string(temp);
      
      // Only use the first word of the trend
      char * first = strtok(temp, " ");
      
      // Ignore trends that have no english characters other than a hash
      if (first == NULL ||
          strncmp(first, "#", TWEETSIZE) == 0)
          continue;
      
      strncpy(trends[trend_index], first, TWEETSIZE);
      //free(temp);
      
      trend_index++;
      // Release references to JSON objects
      json_decref(text);
      json_decref(json_trend_obj);
    }

    // Release references to JSON objects <- causes memory corruption
    // ("corrupted double-linked list")
    //json_decref(json_trends_array);
    //json_decref(first_object);
    //json_decref(root);

    //free(line);
    
    // Return the number of trends read
    return trend_index;
  }
  
  // Ran out of input. Just return 0
  return 0;
} // read_trends

// From the starter code for the Twitter lab
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
    if(!root) {
      fprintf(stderr, "Invalid json line encountered while reading tweets\n");
      continue;
    }
  
    // Skip over lines that aren't JSON objects
    if(!json_is_object(root)) {
      fprintf(stderr, "Invalid json object encountered while reading tweets\n");
      json_decref(root); // TODO failed json_loads doesn't allocate memory
      continue;
    }
  
    // Get the text of the tweet
    json_t* text = json_object_get(root, "text");
  
    // If there was no text, skip this tweet
    if(!json_is_string(text)) {
      fprintf(stderr,
                "Invalid json string encountered while reading trends\n");
      json_decref(root);
      continue;
    }
  
    // Get the string out of the JSON text value
    const char* json_text = json_string_value(text);
  
    // Got a tweet! Copy just the tweet text to an allocated buffer
    char* tweet_text = (char*)Malloc(sizeof(char) * (line_length + 1),
                                      "tweet_text");
    strncpy(tweet_text, json_text, TWEETSIZE);
    
    // Release this reference to the JSON object
    //json_decref(root);
    
    // Return the result
    return tweet_text;
  }
  fprintf(stderr, "Could not fetch any more tweets\n");
  // Ran out of input. Just return NULL
  return NULL;
}
/******************
 * CUDA functions *
 ******************/

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
                                      char * gpu_word_counts_per_tweet,
                                      int num_trends) {
  int trend_index =  threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
  if (trend_index < num_trends) {
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
                                     int * weights,
                                     int word_count,
                                     int num_trends) {
  int trend_index =  threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
  if (trend_index < num_trends) {
    int num_correlated_words = 0;
    int num_tweets_w_trend = gpu_tweets_in_trends[trend_index];
    // Ignore trends with no tweets
    if (num_tweets_w_trend == 0) {
      correlated_words[word_count * trend_index + num_correlated_words] =
        END_OF_TWEET;
      return;
    }
    for (int word_index = 0; word_index < word_count; word_index++) {
      int count = gpu_word_counts_per_trend[word_count * trend_index +
                                            word_index];
      double freq_in_trend = count / (double) num_tweets_w_trend;
      double total_freq =  total_word_counts[word_index] / (double) word_count;
      if (freq_in_trend > CORRELATION_FACTOR * total_freq && count > 1) {
        // got a correlated word! Record it
        correlated_words[word_count * trend_index + num_correlated_words] =
          word_index; // And its weight:
        weights[word_count * trend_index + num_correlated_words] = count;
        num_correlated_words++;
      }
      // Signify the end of correlated words array
      correlated_words[word_count * trend_index + num_correlated_words] =
        END_OF_TWEET;
    }
  }
}

/************************************
 * Error checking wrappers for CUDA *
 ************************************/

// Error checking wrapper for mallocing
void * CudaMalloc(size_t size, char * error_message) {
  void * ptr;
  if(cudaMalloc(&ptr, size) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate %s on GPU\n", error_message);
    exit(1);
  }
  return ptr;
}

/*
 * Error-checking wrapper for cudaMemcpy
 * error message should include what's being copied and a direction preposition
 * e.g "tweets to" or "trends from" 
 */
void CudaMemcpy(void * destination, void * source, size_t size,
                enum cudaMemcpyKind direction, char * error_message) {
  if(cudaMemcpy(destination, source, size, direction) != cudaSuccess) {
    fprintf(stderr, "Failed to copy %s the GPU\n", error_message);
    exit(1);
  }
}

// Error-checking wrapper for cudaMemset to 0
void CudaZeroOut(void * ptr, size_t size, char * error_message) {
  if(cudaMemset(ptr, 0, size) != cudaSuccess) {
    fprintf(stderr, "Failed to zero out %s on GPU\n", error_message);
    exit(1);
  }
}

// Error-checking wrapper for cudaDeviceSynchronize()
void CudaDeviceSynchronize(char * error_message) {
  if (cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "Failed to sync after %s\n", error_message);
    exit(1);
  }
}
