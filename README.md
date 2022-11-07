# Sentiment-Analysis

Using the Sentiment140 dataset, we aim to predict the polarity of the person tweeting.

A portion of the dataset is shown below.

![My Image](https://github.com/atiyeh2016/Sentiment-Analysis/blob/main/Sentiment%20Analysis/Part_of_Dataset.png)

Using glove42b, a word embedding is related to each word. This embedding maps each word to a feature vector of 300 dimensions.

The structure of the model is as follows:

The input is fed into a one-sided LSTM network with one layer and a hidden dimension of 150. Then, the output is sent to a linear layer with three outputs. Finally, the softmax function is applied to the output of the linear layer and the final output is created.

