import pandas as pd
import matplotlib.pyplot as plt
import csv

reviews_len = []
with open('training.cleaned.csv') as csvfile:
    tweets = csv.reader(csvfile)
    for tweet in tweets:
        reviews_len.append(len(tweet[1]))
        
pd.Series(reviews_len).hist()
plt.show()
pd.Series(reviews_len).describe()