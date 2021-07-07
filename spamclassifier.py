# Importing the libraries
import pandas as pd 
import re
import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
messages=pd.read_csv("./SMSSpamCollection", sep="\t", names=['label', 'message'])

print(messages.head(10))

# Data cleaning and preprocessing
ps=PorterStemmer()
wordnet=WordNetLemmatizer()
corpus=[]
for i in range (0, len(messages)):
    review=re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review=" ".join(review)
    corpus.append(review)

print(corpus)

# Creating the bag of words model
cv=CountVectorizer(max_features=3000)
X=cv.fit_transform(corpus).toarray()

print(X)

y=pd.get_dummies(messages['label'])
y=y.iloc[:, 1].values # done to avoid the dummy variable trap

print(y)

# splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Using the Naive Bayes classifier on the dataset
classifier=MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred=classifier.predict(X_test)

print(y_pred)

# Making the confusion matrix and the accuracy score
cm=confusion_matrix(y_test, y_pred)
acc=accuracy_score(y_test, y_pred)
print("confusion matrix \n", cm)
print("accuracy score: ",acc)