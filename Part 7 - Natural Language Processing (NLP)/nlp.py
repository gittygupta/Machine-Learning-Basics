# Natural Language Processing

'''
csv (comma separated variables)
tsv (tab separated variables)
Here we use a tsv file because in the reviews itself we will have commas. Thus comma won't be a 
delimiter.
'''

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
'''
quoting = 3 ignores the double quotes as separators
'''
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
# 1

'''
We will actually take up only the words relevant to that review, and remove the unimportant variables
i.e, a, an, the, punctuations, maybe numbers etc... And then make our 'bag of words' model.
Example - In the 1st review we have the word "loved". For such words we will use STEMMING, which will
convert this word to "love" i.e, we will stem words such as 'loved', 'loving' into 1 word 'love'.
Basically we will look for a very few relevant words in the review and reduce the number of words.
And we will have the reviews only in the lower texts and get rid of the caps. Then we will apply 
tokenisation.

After the pre-processing of the texts what tokenisation does is it will split all the different reviews
into different words (relevant ones) and then make columns of the separated relevant words and count
how many times that particular word comes up for each review. This is called a SPARSE MATRIX.

"re" is the library with efficent functions to clean texts
"review" -> cleaned review of each of the original reviews
"sub('[^a-zA-Z]', ' ', dataset.values[0, 0])" removes all other stuff except all the capital and small
letters from that particular str. And that ' ' makes sure that the spaces arent removed between words

"dataset.values[0, 0]" is equivalent to "dataset['Review'][0]"
The second method above is easier to find when we have a large dataset and know the column names

In the 1st text we can see that the 3 dots are removed
'''
import re
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review = review.lower()


# Removing all the useless words
# 2

'''
like articles, prepositions, and keeping only relevant nouns
"stopwords" list of nltk library consists only of those irrelevant words that need to be removed
'''
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
review = review.split()
review = []
'''removing the irrelevant words by going through the list
'''
#for word in rev:
#    if not word in set(stopwords.words('english')):
#        review.append(word)

'''The above 3 lines can be written in 1 line after splitting as
'''
review = [word for word in review if not word in set(stopwords.words('english'))]


# Stemming
# 3

'''
Here we keep only the root of the word i.e, for loved, loving etc... we keep only love
We obviously will apply stemming in 1 single word
'''
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
'''Thus we see that "loved" has become "love" now
In the above line we did 2 things at the same time actually. So while execution we won't run the line
at the upper part of the code that only uses stopwords but the just above one which combines both
'''

# Combining of the new formed review back into a string
# 4
review = ' '.join(review)




# Combining all the above 4 steps into 1 for loop to go through all the reviews
# Full text cleaning process
'''In NLP a corpus is a list of cleaned/preprocessed texts
'''
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):    
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    '''
    Thus in the corpus list we see all the reviews have been shortened
    Thus preprocesing of the texts is almost complete
    Next we will remove all the words that appear rarely because there are certain things
    like people's names and stuff which are still not relevant at all.
    '''

# Creating the BAG OF WORDS model
'''
We will take all the unique different words present in the 1000 reviews and make a column for each word
We of course won't make 2 different columns for a single word coming out in 2 different reviews.
We will count how many times a particular word (of the column) appears in each review.
We technically will have mostly 0's in most of the cells cz each review made by each person has some
unique words in it.

The matrix we make is called the sparse matrix and as we have a lot of zeroes its called sparsity of 0.
Then we will train or ML model to make some correlation between "which word is present and how many
number of times" and, "the result (0 or 1) i.e, good or bad review" 

Simply what we are doing here is a classification modelling (binary outcome)

Our matrix of features will be the different column of different words and the dependent variable will
be the result (0/1)

Creation of bag of words model will be done using the process of tokenisation

CountVectorizer actually has everything available for cleaning texts which we just did.
But the manual method gives us more control over which stuff to remove and all.
'''

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, [1]].values
'''
CountVectorizer has got the max_features parameters that allow it to keep only the most frequent words
and remove the less frequent one's.

We have got 1565 different words. We can set the max_features to remove the least frequent 65 words

We can reduce the number of features even more using Dimensionality reduction covered later.
This is called reduction of sparsity
'''


# Training the classification model
# Last step
'''
We can experiment with models but out of experience and tests the most commonly used classification
models used in NLP are Naive Bayes Classification or the Decision Tree Classification
We will use naive bayes here.
'''
# Splitting into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes Classification model to the training set
from sklearn.naive_bayes import GaussianNB 
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Making the CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
The model made 55+91 corrections
'''















































