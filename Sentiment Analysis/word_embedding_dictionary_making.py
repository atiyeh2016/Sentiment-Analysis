import numpy as np
import pickle

sum_v = 0
embeddings_dict = {}
with open("glove.42B.300d.txt", 'r', encoding="utf-8") as f:
    print("open glove42B")
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        sum_v += vector
        embeddings_dict[word] = vector
        
a_file = open("vectorized_words.pkl", "wb")
pickle.dump(embeddings_dict, a_file)
a_file.close()

mean_v = sum_v/len(embeddings_dict)
zero_vector = embeddings_dict["0"]

b_file = open("mean_of_vectors.pkl", "wb")
pickle.dump((mean_v,zero_vector), b_file)
b_file.close()
