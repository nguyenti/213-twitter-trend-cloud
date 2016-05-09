import matplotlib.pyplot as plt
from wordcloud import WordCloud
from time import sleep

fn = "clouds.txt"

if __name__ == '__main__':
    # map of the form trend : map of word:counts
    trend_clouds = {}
    with open(fn, "r") as fp:
        trend = fp.readline()
        while(trend):
            # read correlated words and weights pairs, ignore the last one
            word_counts = fp.readline()
            word_counts = word_counts.split(',')[:-1]
            tuples = [(x.split(':')[0], 
                       int(x.split(':')[1])) for x in word_counts]
            if trend in trend_clouds:
                for k,v in tuples:
                    if k in trend_clouds[trend]:
                        trend_clouds[trend][k] = trend_clouds[trend][k] + v
                    else:
                        trend_clouds[trend][k] = v
            else: # add new entry for this trend
                trend_clouds[trend] = dict(tuples)
            trend = fp.readline()
    # display word clouds
    plt.ion()
    #plt.figure()       
    for trend in trend_clouds.iterkeys():
        wc = WordCloud().generate_from_frequencies(trend_clouds[trend].items())
        plt.figure()
        plt.imshow(wc)
        plt.axis("off")
        plt.pause(2.0) # delay for 2 seconds 
        
    # infinite loop so images stay alive
    while True:
        plt.pause(.05)
       
