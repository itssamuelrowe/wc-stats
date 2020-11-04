import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import math
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.colors as colors
from wordcloud import WordCloud, STOPWORDS
from datetime import datetime
import warnings
from scipy import stats
import nltk
from nltk import FreqDist
import spacy
from nltk.corpus import stopwords
import en_core_web_sm
import pandas_profiling
import matplotlib as mpl

warnings.filterwarnings('ignore')

pd.set_option("display.max_colwidth", 200)

df = pd.read_csv('./m31blackfull.csv')
# TO REMOVE UNWANTED WORDS
# remove unwanted characters, numbers and symbols
df['REVIEW'] = df['REVIEW'].str.replace("[^a-zA-Z#]", " ")

nltk.download('stopwords')
stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

# remove short words (length < 3)
df['REVIEW'] = df['REVIEW'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

# remove stopwords from the text
reviews = [remove_stopwords(r.split()) for r in df['REVIEW']]

# make entire text lowercase
reviews = [r.lower() for r in reviews]

#TOKENISATION AND LEMMATISATION
nlp = en_core_web_sm.load(disable=['parser', 'ner'])
def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
  output = []
  for sent in texts:
    doc = nlp(" ".join(sent)) 
    output.append([token.lemma_ for token in doc if token.pos_ in tags])
  return output

tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
reviews_2 = lemmatization(tokenized_reviews)
print(reviews_2[1]) # print lemmatized review

reviews_3 = []
for i in range(len(reviews_2)):
  reviews_3.append(' '.join(reviews_2[i]))

df['reviews_2'] = reviews_3
df1=pd.DataFrame(df)
df.head()
df.info()

pandas_profiling.ProfileReport(df)

ListWords=[reviews_3]

# All Words
def Bag_Of_Words(ListWords):
    all_words = []
    for m in ListWords:
        for w in m:
            all_words.append(w.lower())
    all_words1 = FreqDist(all_words)
    return all_words1


all_words4 = Bag_Of_Words(ListWords)
ax = plt.figure(figsize=(150,100))
# Generate a word cloud image
wordcloud = WordCloud(background_color='white',max_font_size=40).generate(' '.join(all_words4.keys()))

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
print("Word Cloud",len(all_words4))



# data preprocessing

# function to plot most frequent terms
def freq_words(x, terms = 10):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()
  # plt.savefig('1.png')

freq_words(df['reviews_2'], 10)

# Total numbers of ratings in the home and kitchen product reviews
plt.figure(figsize = (10,6))
sns.countplot(df['RATING'])
plt.title('Total Review Numbers for Each Rating', color='r')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.show()
# plt.savefig('2.png')

# Customer totals for each rating class
df['RATING'].value_counts()

plt.figure(figsize = (10,6))

df.groupby('RATING').RATING.count()
df.groupby('RATING').RATING.count().plot(kind='pie',autopct='%1.1f%%',startangle=90,explode=(0,0.1,0,0,0),)

#data=review_df.copy()
word_count=[]
for s1 in df.reviews_2:
    word_count.append(len(str(s1).split()))

plt.figure(figsize = (8,6))

sns.boxplot(x="RATING",y=word_count,data=df)
plt.xlabel('Rating')
plt.ylabel('Review Length')
plt.show()
# plt.savefig('3.png')

#Since there are outliers in the above boxplot we are not able to clearly visualize.So remove the outliers 
plt.figure(figsize = (8,6))

sns.boxplot(x="RATING",y=word_count,data=df,showfliers=False)
plt.xlabel('Rating')
plt.ylabel('Review Length')
plt.show()
# plt.savefig('4.png')

df.loc[(df['RATING'] == 1.0) | (df['RATING'] == 2.0), 'RCAT'] = 'BAD'
df.loc[(df['RATING'] == 4.0) | (df['RATING'] == 5.0), 'RCAT'] = 'GOOD'
df.loc[(df['RATING'] == 3.0), 'RCAT'] = 'NEUTRAL'
df.head()
# Total numbers of ratings in the home and kitchen product reviews
plt.figure(figsize = (8,6))
sns.countplot(df['RCAT'])
plt.title('Total Review Numbers for Each Rating Class', color='b')
plt.xlabel('Rating Class')
plt.ylabel('Number of Reviews')
plt.show()
# plt.savefig('5.png')

# Customer totals for each rating class
df['RCAT'].value_counts()

##########################################
## PLOT DISTRIBUTION OF REVIEW LENGTH   
##########################################
plt.figure(figsize = (15,8))

review_length = df["REVIEW"].dropna().map(lambda x: len(x))
plt.figure(figsize=(12,8))
review_length.loc[review_length < 2000].hist()
plt.title("Distribution of Review Length")
plt.xlabel('Review length')
plt.ylabel('Number of Reviews')

##########################
## TEXT LENGTH
########################### 

df1 = df.query('RATING == 1.0')
words=df1.reviews_2
ListWords=[words]

#All Words
def Bag_Of_Words(ListWords):
    all_words = []
    for m in ListWords:
        for w in m:
            all_words.append(w.lower())
    all_words1 = FreqDist(all_words)
    return all_words1

all_words4 = Bag_Of_Words(ListWords)
ax = plt.figure(figsize=(15,10))
# Generate a word cloud image
wordcloud = WordCloud(background_color='black',max_font_size=40).generate(' '.join(all_words4.keys()))

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
# plt.save('6.png')
plt.axis("off")
print("Word Cloud",len(all_words4))

df1 = df.query('RATING == 2.0')
words=df1.reviews_2
ListWords=[words]

#All Words
def Bag_Of_Words(ListWords):
    all_words = []
    for m in ListWords:
        for w in m:
            all_words.append(w.lower())
    all_words1 = FreqDist(all_words)
    return all_words1

all_words4 = Bag_Of_Words(ListWords)
ax = plt.figure(figsize=(15,10))
# Generate a word cloud image
wordcloud = WordCloud(background_color='black',max_font_size=40).generate(' '.join(all_words4.keys()))

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
# plt.save('7.png')
plt.axis("off")
print("Word Cloud",len(all_words4))

df1 = df.query('RATING == 3.0')
words=df1.reviews_2
ListWords=[words]

#All Words
def Bag_Of_Words(ListWords):
    all_words = []
    for m in ListWords:
        for w in m:
            all_words.append(w.lower())
    all_words1 = FreqDist(all_words)
    return all_words1

all_words4 = Bag_Of_Words(ListWords)
ax = plt.figure(figsize=(15,10))
# Generate a word cloud image
wordcloud = WordCloud(background_color='black',max_font_size=40).generate(' '.join(all_words4.keys()))

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
# plt.save('8.png')
plt.axis("off")
print("Word Cloud",len(all_words4))

df1 = df.query('RATING == 4.0')
words=df1.reviews_2
ListWords=[words]

#All Words
def Bag_Of_Words(ListWords):
    all_words = []
    for m in ListWords:
        for w in m:
            all_words.append(w.lower())
    all_words1 = FreqDist(all_words)
    return all_words1

all_words4 = Bag_Of_Words(ListWords)
ax = plt.figure(figsize=(15,10))
# Generate a word cloud image
wordcloud = WordCloud(background_color='black',max_font_size=40).generate(' '.join(all_words4.keys()))

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
# plt.save('9.png')
plt.axis("off")
print("Word Cloud",len(all_words4))

df1 = df.query('RATING == 5.0')
words=df1.reviews_2
ListWords=[words]

#All Words
def Bag_Of_Words(ListWords):
    all_words = []
    for m in ListWords:
        for w in m:
            all_words.append(w.lower())
    all_words1 = FreqDist(all_words)
    return all_words1

all_words4 = Bag_Of_Words(ListWords)
ax = plt.figure(figsize=(15,10))
# Generate a word cloud image
wordcloud = WordCloud(background_color='black',max_font_size=40).generate(' '.join(all_words4.keys()))

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
# plt.save('10.png')
plt.axis("off")
print("Word Cloud",len(all_words4))