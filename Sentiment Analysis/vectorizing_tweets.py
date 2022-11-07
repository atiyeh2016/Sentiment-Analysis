import csv
import pickle
import time

a_file = open("vectorized_words.pkl", "rb")
embeddings_dict = pickle.load(a_file)
a_file.close()

b_file = open("mean_of_vectors.pkl", "rb")
mean_v = pickle.load(b_file)
b_file.close()
c = 0

with open('training.cleaned.csv', 'r', newline='') as csvfile:
    print("training.cleaned")
    wordy_tweets = csv.reader(csvfile)

    with open('training.word_embedded.csv', 'w', newline='') as csvfile:
        print('training.word_embedded')
        fieldnames = ['Polarity','Vector']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        
        t1 = time.time()
        for wordy_tweet in wordy_tweets:
            c += 1
            if c%100 == 0:
                print(c)
                print(time.time()-t1)
            
            if  wordy_tweet[0] != "polarity":
                vectorized_tweet = {'Polarity': wordy_tweet[0]}
                tweet = wordy_tweet[1]
                vocabs = tweet.split()
                
                vectors = []
                for vocab in vocabs:
                    try:
                        vectors.append(embeddings_dict[vocab.lower()])
                    except:
                        vectors.append(mean_v)
                    
                for n in range(280-len(vocabs)):
                    vectors.append(embeddings_dict["0"])
                
                vectorized_tweet = {'Vector': vectors}
                writer.writerow(vectorized_tweet)
#        