# 213-twitter-trend-cloud
   
## How to run the program
### To get the data
If you have not set up a developer account with twitter, follow the steps from Charlie Curtsinger's CSC 213 [Twitter Lab](http://www.cs.grinnell.edu/~curtsinger/teaching/2016S/CSC213/labs/twitter/).

Once you have set that up and received the OAuth signature, you will need the *Authorization header* from the OAuth signing results for both the [Twitter sample stream](https://dev.twitter.com/streaming/reference/get/statuses/sample) and the [Twitter trends stream](https://dev.twitter.com/rest/reference/get/trends/place). 

To run our program, run
```
make && make run TWEET_OAUTH="'<INSERT YOUR TWITTER AUTHORIZATION HERE>'" TRENDS_OAUTH="'<INSERT YOUR TRENDS AUTHORIZATION HERE>'"
```


### To produce the word cloud
We used an external library in order to produce the word cloud. You can follow the instructions [here](https://github.com/amueller/word_cloud) to get the setup or just run:

```
pip install wordcloud
```

In order to produce the word cloud, run:

```
python cloud_maker.py
```

Save the cloud, if you wish.

## Example:
1. Get OAuth for both the [Twitter sample stream](https://dev.twitter.com/streaming/reference/get/statuses/sample) and the [Twitter trends stream](https://dev.twitter.com/rest/reference/get/trends/place)
2. Run
	```
	make && make run make && make run TWEET_OAUTH="'<INSERT YOUR TWITTER AUTHORIZATION HERE>'"
	```
3. Install wordcloud (need to do this only one time)
4. Run
	```
	python cloud_maker.py
	```
5. Save the cloud(s)

Example cloud output:

![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Word cloud for ______"

## Future Optimizations
1. Sort and use binary search for words
2. Make get_trend_word_counts go across rows (one thread per tweet), not for every trend
