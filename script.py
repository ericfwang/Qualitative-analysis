# ---------------------------------------------------------------------------------------------------------------------
# Author: Eric Wang
# Source:
#  1. IMDb Reviews: https://www.kaggle.com/atulanandjha/imdb-50k-movie-reviews-test-your-bert?select=train.csv
# ---------------------------------------------------------------------------------------------------------------------
# The goal of this project is to use content analysis/sentiment analysis methods to predict the sentiment of online
# movie reviews. IMDb is one of the most prominent sources for online movie reviews. I thoroughly clean the text review
# data before stemming and applying a bag-of-words model. I then use logistic regression to predict the sentiment
# of the reviews.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn import metrics

from nltk import word_tokenize
from nltk.stem import PorterStemmer

DATAROOT = '/Users/ericwang/Documents/GitHub datasets/Sentiment analysis/'
EXPORT = '/Users/ericwang/Documents/GitHub/Sentiment analysis/'

# Import the data and convert the labels
reviews = pd.read_csv(DATAROOT + 'imdb_reviews.csv')
reviews['sentiment'] = np.where(reviews['sentiment'] == 'neg', 0, 1)

# Check for dataset imbalance and then sample it
print(reviews['sentiment'].value_counts() / len(reviews))
sample_size = 8000
reviews = reviews.sample(n=sample_size, random_state=1)

# Tokenize the review and filter to purely alphabetical words
tokens = [word_tokenize(review) for review in reviews.text]
print('Original tokens: ', tokens[0])

cleaned_tokens = [[word for word in item if word.isalpha()] for item in tokens]
print('Cleaned tokens: ', cleaned_tokens[0])

# Add on length of review after these filtering steps
len_tokens = []
for i in range(len(cleaned_tokens)):
     len_tokens.append(len(cleaned_tokens[i]))

# Stem the words using the Porter stemming algorithm
porter = PorterStemmer()
stemmed_tokens = [[porter.stem(word) for word in review] for review in cleaned_tokens]
print('Stemmed tokens: ', stemmed_tokens[0])

# Piece back the tokens
processed_reviews = [' '.join(review) for review in cleaned_tokens]
print('Processed reviews: ', processed_reviews[0])

# Translate text values to numeric using BOW while filtering out common stop words in movie reviews
my_stop_words = ENGLISH_STOP_WORDS.union(['film', 'theater', 'movie', 'saw', 'title', 'cinema', 'picture'])
vect = CountVectorizer(max_features=1500, stop_words=my_stop_words)
vect.fit(processed_reviews)

# Finalize the data
bag_of_words = vect.transform(processed_reviews)
for_model = pd.DataFrame(bag_of_words.toarray(), columns=vect.get_feature_names())
for_model['length'] = len_tokens
print(for_model.head())

# Create a logistic regression model
x_train, x_test, y_train, y_test = train_test_split(for_model, reviews['sentiment'], test_size=0.2)

model = LogisticRegression(penalty='l2', C=0.5, solver='liblinear')
model.fit(x_train, y_train)
predictions = model.predict(x_test)
probabilities = model.predict_proba(x_test)
model.score(x_test, y_test)

# Plot ROC curve and label with the AUC
y_predictions = model.predict_proba(x_test)[::,1]
false_pr, true_pr, _ = metrics.roc_curve(y_test, y_predictions)
auc = metrics.roc_auc_score(y_test, y_predictions)

plt.plot(false_pr, true_pr, label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
plt.savefig(EXPORT + 'roc.png')