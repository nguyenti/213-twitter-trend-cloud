#include "util.c"

#define TREND_FETCH_TIME (5 * 60 * 1000) // should be 5 min
#define THREADS_PER_BLOCK 64


char* read_tweet(FILE * stream);
size_t read_trends(char ** trends, FILE * file);

//TODO: RENAME trend_maps, gpu_matrix, gpu_hashed_words


/* 
 * Populate a NUMTWEETS x word_count matrix with word counts per tweet
 *
 * @post gpu_matrix[i * word_count + j] == # of occurrences of word j in tweet i
 *
 * gpu_tweets - the compressed representation of tweets
 * gpu_hashed_words - the compressed representation of words
 * TODO: these words aren't actually hashed, are they?
 * gpu_matrix - the resulting matrix of word counts per tweet 
 * word_count - the number of distinct words in the current batch
 */
__global__ void compute_word_containment(int * gpu_tweets, 
                                         int * gpu_hashed_words,
                                         char * gpu_matrix,
                                         int word_count);

/* 
 * Populate a NUMTRENDS x word_count matrix with word counts per trend
 *
 * @post trend_maps[i * word_count + j] == number of occurrences of word j in
 *                                         tweets that contain trend i
 *
 * trend_maps - a NUMTRENDS x word_count matrix, denoting word counts per trend
 * gpu_tweets_in_trends - array of number of tweets containing each trend 
 * gpu_trends - the compressed representation of trends
 * word_count - the number of distinct words in the batch
 * gpu_matrix - the matrix of word counts per tweet 
 */
__global__ void get_trend_word_counts(int * trend_maps,
                                      int * gpu_tweets_in_trends,
                                      int * gpu_trends,
                                      int word_count,
                                      char * gpu_matrix);

/*
 * Populate the matrix of words correlated with each trend


TODO: maybe combine with get_trend_word_counts? the resulting matrix is very
similar, has same dimensions

 * trend_maps - word counts for each trend
 * gpu_tweets_in_trends - number of tweets in a trend
 * total_word_counts - word counts for all words in all tweets
 * correlated_words - output, what's correlated
 * word_count - the number of distinct words in the batch
 */
__global__ void get_correlated_words(int * trend_maps,
                                     int * gpu_tweets_in_trends,
                                     int * total_word_counts,
                                     char * correlated_words,
                                     int word_count);


// Error-checking wrapper for cudaMalloc
void * CudaMalloc(size_t size, char * error_message) {
  void ** ptr = NULL;
  if(cudaMalloc(ptr, sizeof(int) * NUMTWEETS * COMPRESSEDLEN)
     != cudaSuccess) {
    fprintf(stderr, "Failed to allocate %s on GPU\n", error_message);
    exit(2);
  }
  return *ptr;
}

// Error-checking wrapper for cudaMemcpy
void CudaMemcpy(void * destination, void * source, size_t size,
                enum cudaMemcpyKind direction, char * error_message) {
  if(cudaMemcpy(destination, source, size, direction) != cudaSuccess) {
    fprintf(stderr, "Failed to copy %s to the GPU\n", error_message);
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
  /*
  int * gpu_tweets;
  if(cudaMalloc(&gpu_tweets, sizeof(int) * NUMTWEETS * COMPRESSEDLEN)
     != cudaSuccess) {
    fprintf(stderr, "Failed to allocate the tweets on GPU\n");
    exit(2);
  }
  */
  int * gpu_trends;
  if(cudaMalloc(&gpu_tweets, sizeof(int) * NUMTRENDS) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate the trends on GPU\n");
    exit(2);
  }
  
  // Topic containment matrix
  char * trend_matrix = (char *)calloc(sizeof(char), // size of a cell
                                       sizeof(char) * NUMTWEETS * NUMTRENDS);
  char * gpu_matrix;
  if(cudaMalloc(&gpu_tweets, sizeof(int) * NUMTWEETS * NUMTRENDS)
     != cudaSuccess) {
    fprintf(stderr, "Failed to allocate the matrix on GPU\n");
    exit(2);
  }
  // word count maps for every trend
  int * trend_maps; //TODO zero out on every iteration

  // Word arrays
  char words[NUMTWEETS*COMPRESSEDLEN][TWEETSIZE];
  int hashed_words[NUMTWEETS*COMPRESSEDLEN]; // TODO we don't need this?
  int total_word_counts[NUMTWEETS*COMPRESSEDLEN];
  int word_count = 0;
  
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
        // TESTING
        
        // Copy trends onto the GPU
        if(cudaMemcpy(gpu_trends, compressed_trends, sizeof(int) * NUMTRENDS,
                      cudaMemcpyHostToDevice) != cudaSuccess) {
          fprintf(stderr, "Failed to copy trends to the GPU\n");
        }
      }
    }
    
    // save tweet by copying
    strncpy(tweets[tweet_count], tweet, TWEETSIZE);

    // TODO: Clean and compress the tweet
    clean_string(tweet);
    compress_str(tweet,
                 compressed_tweets[tweet_count],
                 words,
                 hashed_words,
                 total_word_counts,
                 &word_count);
    free(tweet);
    
    // TESTING
    printf("tweet #%d: %s\n", tweet_count, tweet);

    if (tweet_count >= NUMTWEETS - 1) {
      // Copy compressed tweets onto the GPU
      if(cudaMemcpy(gpu_tweets, compressed_tweets,
                    sizeof(int) * NUMTWEETS * COMPRESSEDLEN,
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy tweets to the GPU\n");
      }

      // Allocate trend maps
      if(cudaMalloc(&trend_maps, sizeof(int) * word_count * NUMTRENDS)
         != cudaSuccess) {
        fprintf(stderr, "Failed to allocate trend_maps on GPU\n");
        exit(2);
      }

      // Zero out trend_maps
      if(cudaMemset(&trend_maps, 0, sizeof(int) * word_count * NUMTRENDS)
         != cudaSuccess) {
        fprintf(stderr, "Failed to zero out trend_maps on GPU\n");
        exit(2);
      }

      // Figure out trends' compressed values (indices in the word array)
      for (int i = 0; i < num_trends; i++) {
        // go through words array to find each trend'd index
        for (int j = 0; j < word_count; j++) {
          if (strcmp(trends[i], words[j]) == 0) {
            compressed_trends[i] = j;
            break;
          }
        }
      }

      // TODO copy compressed_trends onto the GPU
      // TODO: copy word_count onto GPU
      // TODO: copy hashed_words onto GPU
      // TODO: copy words onto GPU
      // char words[NUMTWEETS*COMPRESSEDLEN][TWEETSIZE];
      // TODO: create gpu_tweets_in_trends on GPU & zero out

      // Zero out gpu_tweets
      
      // Make tweets x words matrix of counts. (similar to the thing below)  
      // TODO: Make an NxK topic containment bit matrix
      compute_word_containment<<<1, NUMTWEETS>>>(gpu_tweets, gpu_hashed_words,
                                                 gpu_matrix, word_count);

      // TODO: That syncing thing
      
      // To make trend_maps, add rows of the tweet X word matrix that correpond
      // to each trend
      // TODO: Get word counts for each tweet with a specific trend
      get_trend_word_counts<<<1, NUMTRENDS>>>(trend_maps,
                                              gpu_tweets_in_trends,
                                              gpu_trends,
                                              word_count,
                                              gpu_matrix);
      
      // TODO: Find word sets correlated with each topic and compute correlation
      //   coefficients
      get_correlated_words<<<1, NUMTRENDS>>>(trend_maps,
                                             gpu_tweets_in_trends,
                                             total_word_counts,
                                             correlated_words,
                                             word_count);

      
      // TODO: Create word clouds with external tools (go to 8 if it doesn’t
      //   work out)
      // TODO: If time allows: Implement weighing words by importance (tf-idf)
      // TODO: If time allows: Explore other uses of the same output data: graph
      //   building, clustering, or term evolution over time
      // TODO: If time allows: Rewrite compress_str to use insertion sort and binary search
      //   maybe with an auxiliary data structure

      // Free stuff: trend_maps
      // Zero out gpu_matrix
      
      
      word_count = 0;
      tweet_count = 0;
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



/* CUDA functions */

__device__ void get_intersect(int *tweet1,int *tweet2, int *intersect){
  intersect = (int*) malloc(sizeof(int) * (COMPRESSEDLEN));
  int i = 0; // tweet1
  int j = 0; // tweet2
  int index_intersect;
  int k = 0; // For the intersection
  while (tweet1[i] != 0) { 
    while (tweet2[j] != 0) {
      if (tweet2[j] == tweet1[i]) {
        index_intersect = 0;
        while (index_intersect < k) {
          // check that we have no repeats in our intersection
          if (tweet1[i] == intersect[index_intersect]) {
            break;
          }
        }
        // if it didn't find the tweet in the intersection, add it
        if (index_intersect == k) {
          intersect[k++] = tweet1[i];
        }
      }
      j++;
    }
    i++;
    j = 0;
  }
}

// N tweets x K trends
// gpu_matrix must be zeroed out
__global__ void compute_word_containment(int * gpu_tweets,
                                         int * gpu_hashed_words,
                                         char * gpu_matrix,
                                         int word_count) {
  int index =  threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
  if (index < NUMTWEETS) {
    int * tweet = gpu_tweets[COMPRESSEDLEN * index];
    for (int i = 0; i < COMPRESSEDLEN && tweet[i] != 0; i++) {
      for (int j = 0; j < word_count; j++) {
        if (tweet[i] == gpu_hashed_words[j]) // should be just  == j
          gpu_matrix[word_count * index + j]++;
        /* //Can't we just say this, since tweet[i] *is* the index of the word
           gpu_matrix[word_count * index + tweet[i]]++;
         */
      }
    }
  }
}

__global__ void get_trend_word_counts(int * trend_maps,
                                      int * gpu_tweets_in_trends,
                                      int * gpu_trends,
                                      int word_count,
                                      char * gpu_matrix) {
  int trend_index =  threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
  if (trend_index < NUMTRENDS) {
    int trend_map_index = trend_index * word_count;
    int trend_word_index = gpu_trends[trend_index];
    for (int i = 0; i < NUMTWEETS; i++) { // for every tweet
      if (gpu_matrix[i * word_count + trend_word_index] > 0) {
        // if the trend is present in the tweet
        for (int j = 0; j < word_count; j++) {
          // get the word counts for all the words in the tweet
          trend_maps[trend_map_index + j] +=
            gpu_matrix[i * word_count + j];
        }
        gpu_tweets_in_trend[trend_index]++;
      }
    }
  }
}

__global__ void get_correlated_words(int * trend_maps,
                                     int * gpu_tweets_in_trends,
                                     int * total_word_counts,
                                     char * correlated_words,
                                     int word_count) {
  int trend_index =  threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
  if (trend_index < NUMTRENDS) {
    for (int i = 0; i < word_count; i++) {
      if (trend_maps[word_count * trend_index + i] /
          (double) gpu_tweets_in_trends[trend_index] >
          CONSTANT * total_word_counts[i] / (double) word_count) {
        correlated_words[word_count * trend_index + i] = 1;
      } else {
        correlated_words[word_count * trend_index + i] = 0;
      }
    }
  }
}
