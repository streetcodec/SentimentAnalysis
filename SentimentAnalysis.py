import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt') 
stopwords = set(stopwords.words('english'))
stopwords.remove('not')

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
nltk.download('averaged_perceptron_tagger') 
nltk.download('wordnet') 
nltk.download('omw-1.4') 
lemmatizer  = WordNetLemmatizer()


#Utility Functions 
def cleanstr(text):
    text= str(text).lower()
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(' \d+', ' ', text)
    return text
def remove_stopwords(text):
  token = word_tokenize(text)
  token_without_stopwords = []
  for words in token:
    if words not in stopwords:
      token_without_stopwords.append(words)
  
  text = " ".join(token_without_stopwords)
  return text
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV 
    else:
        return wordnet.NOUN 
def lemmatize(text):
  token = word_tokenize(text)
  word_pos_tags = nltk.pos_tag(token)
  token_with_lemmatizer = []
  for idx, tag in enumerate(word_pos_tags):
    token_with_lemmatizer.append(lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1])))
  
  text = " ".join(token_with_lemmatizer)
  return text 


# data prep
df = pd.read_csv("1429_1.csv",usecols=['id','reviews.rating', 'reviews.text'], encoding = 'utf8')
df.head()

df = df.fillna(method = "ffill",axis = 0) # filling missing values

sentiment = {
            1: -1,
            2: -1,
            3: 0,
            4: 1,
            5: 1} # -1 is negative sentiments, for 1 and 2 star rating, 0 is neutral sentiment and 1 is positive sentiment.

df['sentiment'] = df['reviews.rating'].map(sentiment) 
df['reviews.text'] = df['reviews.text'].apply(cleanstr)
df['reviews.text'] = df['reviews.text'].apply(remove_stopwords)
df['reviews.text'] = df['reviews.text'].apply(lemmatize)
df.head()

# Model Implementation 

index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= 0.8]
test = df[df['random_number'] > 0.8]
Y_train = train['sentiment'] 
Y_test = test['sentiment'] 


cv = CountVectorizer()
X_train  = cv.fit_transform(train['reviews.text']) 
X_test = cv.transform(test['reviews.text']) 
X_train =  X_train.toarray() 
X_test = X_test.toarray() 

model = Sequential() 
model.add(Dense(units = 16, activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dense(units = 8, activation = 'relu' ))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])
history = model.fit(X_train, Y_train, epochs = 30)

#result 

model.summary()
test_loss, test_acc  = model.evaluate(X_test,Y_test)
print(test_loss)
print(test_acc)
