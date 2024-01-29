# Tweet-emotion-recognition-with-TensorFlow
Tweet Emotion Recognition 
Introduction 
-In this project, we're going to use TensorFlow to create and train a Recurrent Neural Network model which will be able to classify tweets into 6 categories according to their emotional tone. 

I- Setup 
! pip install datasets

This line of code uses pip (the Python package installer) to install the nlp package which includes datasets, preprocessing and evaluation tools and pre-trained models. 

II- Importing data 
i- Import the emotions dataset from the Hugging Face Datasets repository 
import datasets 
dataset = datasets.load_dataset("emotions")

• datasets.load_dataset() will download the emotion dataset and return a reference to it as a DatasetDict object.

• The DatasetDict object is a Python dictionary-like object that is returned by the Hugging Face Datasets library. 
It contains the dataset in the form of several splits, such as “train”, “validation” and “test”.
Note: each split is represented as a dataset object.

ii- Save the three subsets or splits into lists of dictionaries 
trainSet = dataset['train']
testSet = dataset['test']
validationSet = dataset['validation']



iii- Extract tweets and labels from a given list of dictionaries 
def getTweet(data):
   tweets = [x['train'] for x in data]
   emotions = [x['label'] for x in data]


• The function getTweet() takes one argument “data”, which is a list of dictionaries.
• It uses list comprehension to extract the text of each tweet and the corresponding sentiment label from  “data” and store them in tweets and emotions lists, respectively.
• The function returns a tuple containing the tweets and emotions lists.



iv- View a tweet and its corresponding sentiment label from the training dataset
trainTweet, trainLabel = getTweet(dataset['train'])
print(trainTweet[0], " ", trainLabel[0])








III- Preprocessing data
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(dataset)


i- tokenization 
A tokenizer is used to split the training data (tweets) into smaller units called tokens , in this case tokens are words. 

• The first line of code creates an instance of the Tokenizer class and assigns it to the tokenizer variable.
• Num_words specifies the maximum number of words to keep in the vocabulary. In this case, we are setting it to 10,000, which means that the tokenizer will keep the 10,000 most frequently occurring words in the training data and discard any less common words. 

Note: This is a way to limit the size of the vocabulary and potentially improve the performance of the neural network.

• The oov_token parameter specifies the token to be used for out-of-vocabulary words. This means that if the tokenizer encounters a word that is not in the vocabulary (i.e. not one of the 10,000 most common words), it will replace that word with the <UNK> token.

ii- vectorization or encoding 
• In the second line of code, the tokenizer’s vocabulary is assigned the top 10,000 common words in the dataset, and a unique integer is assigned to each word in the vocabulary, based on its frequency. 

Note: Words that occur more frequently will be assigned lower indices, and less frequent words will be assigned higher indices. 



IV- Padding and truncating sequences 
i- display the lengths of all tweets in the training set to decide on the length of sequences.
lengths = [len(t.split(' ') for t in tweets] 
plt.hist(lengths)
plt.show()





From the above histogram, we notice that very few tweets have a length greater than 50, and so the fixed size of all sequences will be 50. 

ii- creating then padding and truncating sequences 
maxLen = 50 
def getSequnces(tokenizer, tweet):
   sequences = tokenizer.texts_to_sequences(tweets)   
   padded_sequences = pad_sequneces(sequences,
                                   truncating='post',
                                   padding='post', 
                                   maxlen=maxLen) 


i- sequences = tokenizer.texts_to_sequences(tweets)   
The tokenizer’s method “texts_to_sequences” will convert every tweet into a sequence (list) of integers, and save all sequences in a list called sequences.  

ii- padding and truncating sequences
• sequences: A list of sequences of integers to be padded. 
• Truncating /padding : A string indicating whether the truncating/padding should be added to the beginning of the sequence (truncating =’pre’ or padding='pre') or the end of the sequence (truncating=’post’ or padding='post').
• maxlen: The maximum length of the padded sequence. If a sequence is shorter than maxlen, it will be padded with zeros at the beginning or end (depending on the value of padding) until it reaches the maximum length. If a sequence is longer than maxlen, it will be truncated so that it only contains the first maxlen elements.

Note: 
The output of getSequences method is a NumPy array with shape (number of sequences, max sequence length), where the max sequence length is 50.


V- Preparing labels 
i- create a list of strings that represents the unique labels 
classes = set(dataset['train'].features['label'].names)


ii- display the number of tweets for each class/ label
plt.hist(labels)
plt.show()


iii- create dictionaries
classToIndex = dict((c, i) for i, c in enumerate(classes))
indexToClass = dict((v, k) for k, v in classToIndex.items())

• The first line of code creates a dictionary that maps each class of the six classes; anger, joy, sadness, love, surprise, and fear, to its corresponding integer index in the vocabulary. 

• The second line of code creates a dictionary that maps each integer to a corresponding class.
Note: classToIndex.items() returns a list of key-value pairs from the dictionary, where each key-value pair is represented as a tuple (key, value).








VI- Creating the model
model = Sequential([
   Embedding(10000, 16, input_length=maxLen),
   Bidirectional(LSTM(20, return_sequences=True)),
   Bidirectional(LSTM(20)),
   Dense(6, activation="softmax")

])


i- Embedding layer: 
This layer maps each word of the top 10,000 most common words to a fixed-size vector of dimension 16. The input_length parameter specifies the length of the input sequence, which is denoted by maxLen and equals 50.
Note: 10,000 is the size of the vocabulary.

ii- first bidirectional LSTM layer: 
It processes the input sequence in both forward and backward directions, which can improve the model's ability to capture long-term dependencies in the data. The 20 parameter specifies the number of LSTM units, and “return_sequences=True” specifies that the layer should return the full sequence of outputs, rather than just the final output. 





iii- Second bidirectional LSTM layer:
 also has 20 units but it only returns the final output of the sequence .

Notes: 
•  The Bidirectional layer is a wrapper layer that takes any RNN layer as an argument and processes the input sequence in both forward and backward directions, which allows it to capture both past and future context when making predictions. 
•  In this case, the RNN layer being wrapped is an LSTM layer.
•  The output of the bidirectional LSTM layer is a sequence of vectors, where each vector represents the hidden state of the LSTM at a particular time step. The sequence length is the same as the input sequence length which is 50, and the dimensionality of each vector is 16. 
iv- Dense 
maps the output of the second bidirectional LSTM layer to a vector of size 6 ,which is the number of classes/ emotions. The softmax activation function normalizes the output vector to represent a probability distribution over the possible classes. 








VII- Compiling the model: 
model.compile(
   optimizer='adam',
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy']
)

i- optimizer: This parameter specifies the optimization algorithm used to update the weights of the neural network during training.
ii- loss: This parameter specifies the loss function that is used to measure the difference between the predicted output of the neural network and the true output during training. 
iii- metrics: This parameter specifies the evaluation metric used to 	monitor the performance of the neural network during training. 



VIII- Training the model: 
model = model.fit(paddedTrainSeq, trainLabels,
                  validation_data=(validationSequences,
                                  validationLabels),
                  epochs=20,
                  callbacks=[EarlyStopping(monitor='val_accuracy',
                                           patience=2)]) 




• paddedTrainSeq, trainLabels specify the training dataset. The fit method uses this data to train the neural network and adjust the weights of the network to minimize the loss function.
• Validation_data specifies a separate validation dataset to evaluate the performance of the neural network during training. 
• Epoch is an iteration during which the neural network processes the entire training dataset, computes the loss function, and updates the weights of the network using backpropagation.
• The EarlyStopping callback is used to stop training early if the validation accuracy does not improve for 3 consecutive epochs.

IX- Evaluating the model: 
testTweets, testLabels = getTweet(testSet)
testLabels = np.array(testLabels)
testSeq = getSequences(tokenizer, testTweets)
evaluatedModel = model.evaluate(testSeq, testLabels)




Test tweets are being sequenced and the labels are converted into numpy arrays so that they’re in the right form to be parameters of the evaluate method.



 
X- Making predictions: 
i = random.randint(0, len(testTweets) - 1)
print('Tweet: ', testTweets[i])
print('Emotion: ', indexToClass[testLabels[i]])
prediction = model.predict(np.expand_dims(testSeq[i], axis=0))
predictedValue = indexToClass[np.argmax(prediction)]
print('Predicted emotion: ', predictedValue)




• The np.expand_dims() function is used to add an extra dimension at position 0 to the ith sequence, which converts it from a one-dimensional numpy array to a two-dimensional numpy array with shape (1, maxLen), where 1  represents  the number of input samples to be generated predictions for. 
• The “axis=0” argument specifies that the new dimension should be added at position 0 of the array.

• This is necessary because the model.predict() method expects its input to be a 2D numpy array.
• Model.predict returns a 2D array of output predictions, where each row corresponds to a single input sample and each column corresponds to a different class or label.  

Note: in this case there is only one row because we only have one input sample to be generated a prediction for, which means that the variable prediction is a 1D numpy array of length 6 which is the number of classes or labels. 

• np.argmax(prediction) returns  the index of the maximum value in the prediction array. 
