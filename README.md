# Sentiment-Analysis

Using the Sentiment140 dataset, the aim is to predict the polarity of the person tweeting.

A portion of the dataset is shown below.

![My Image](https://github.com/atiyeh2016/Sentiment-Analysis/blob/main/Sentiment%20Analysis/Part_of_Dataset.png)

Using glove42b, a word embedding which maps each word to a feature vector of 300 dimensions is used.

The structure of the model is as follows:  

&emsp;The input is fed into a one-sided LSTM network with one layer and a hidden dimension of 150.  
&emsp;The output is sent to a linear layer with three outputs.  
&emsp;Finally, the softmax function is applied to the output of the linear layer and the final output is created.

To compare the performance, the pre-trained BERT network was fine-tuned using the Sentiment140 dataset and a tokenizer to convert each sentence to pre-defined tokens and IDs.

