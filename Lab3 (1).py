#!/usr/bin/env python
# coding: utf-8

# In[127]:


import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


# In[57]:


nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')


# In[58]:


#1A -- Wordcloud
def get_cloud(data):
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens=[]
    for i in data:
        temp_list=[word for word in i.split() if word not in stop_words]
        tokens.append(" ".join(temp_list))


    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()

    # TF-IDF for the new corpus
    tfidf_matrix = vectorizer.fit_transform(tokens)


    ### Word Cloud

    # Use the term-document matrix (tf-idf) to count terms
    term_freq = tfidf_matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    term_count = dict(zip(terms, term_freq))

    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, max_words=40, background_color="white").generate_from_frequencies(term_count)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


# In[59]:


df = pd.read_csv("/Users/emilyhenderson/Downloads/TweetDataLab3.csv", encoding = 'unicode_escape')
df.head()


# In[53]:


get_cloud(df['Content'])


# In[77]:


#1B -- Preprocessing of the data
tweetdf = pd.read_csv("/Users/emilyhenderson/Downloads/TweetDataLab3.csv")
content = df['Content'].dropna().astype(str).tolist()
# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
# TF-IDF for the new corpus
tfidf_matrix = vectorizer.fit_transform(content)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

preprocessed_text = [preprocess_text(doc) for doc in content]


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_text)
term_freq = tfidf_matrix.sum(axis=0).A1
terms = vectorizer.get_feature_names_out()
term_count = dict(zip(terms, term_freq))
print('Sum of terms',len(vectorizer.vocabulary_))


# In[ ]:


#1C
#Based on the document term matrix, I have 9637 terms in the cleaned text data now. 


# In[79]:


#1D -- Latent Semantic Analysis
lsa = TruncatedSVD(n_components=10)
lsa_tfidf = lsa.fit_transform(tfidf_matrix)
words_df = pd.DataFrame(lsa_tfidf, columns=['Component1','Component2', 'Component3', 'Component4', 'Component5', 'Component6', 'Component7', 'Component8', 'Component9', 'Component10'])


# In[81]:


print(words_df)


# In[89]:


#2A -- Sentiment Analysis
# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
scores = df['Content'].dropna().astype(str).apply(lambda txt: analyzer.polarity_scores(txt))

# You can take a look at the scores
print([i for i in scores])

# Create new columns to add the scores into data
for i in [i for i in scores.index]:
    df.loc[i,'pos_score']=scores[i].get('pos')
    df.loc[i,'neg_score']=scores[i].get('neg')
    df.loc[i,'neu_score']=scores[i].get('neu')
    df.loc[i,'comp_score']=scores[i].get('compound')


# In[91]:


#2B -- Histogram to visualize sentiment scores
plt.hist(df['comp_score'],alpha=0.7, color='grey',edgecolor='black',label="Overall")
plt.hist(df['pos_score'],alpha=0.9, color='orange',edgecolor='black',label="Positive")
plt.hist(df['neu_score'],alpha=0.5, color='green',edgecolor='black',label="Neutral")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.title("Overall Sentiment")
plt.legend()
plt.show()


# In[119]:


#2C -- Side by side boxplots
plt.figure(figsize=(20, 10))
sns.boxplot(x='comp_score', hue='Expressed_emotion', data = df, orient = 'horizontal')
plt.show()


# In[ ]:


#2D -- Based on the histogram and the boxplots, the most common emotions expressed were happy, positive ones. However, there were relatively
#high worry scores as well. The histogram shows that the majority of the tweets' content was positive, although a decent ratio of the tweets
#were neutral as well. The boxplots show that the most commonly-expressed emotions were eager, fun, love, happy, and relief, which lines
#up with the histogram's portrayal of overwhelmingly positive tweet content. 


# In[109]:


#3A -- Create new dataframe with sentiment scores and LSI concepts
combined_df = pd.concat([df[['pos_score', 'neg_score', 'neu_score', 'comp_score']], words_df], axis=1)
combined_df.head() 


# In[161]:


#3B -- Apply PCA to reduce the combined data
# Normalize the data
scaler = StandardScaler()
data_std = scaler.fit_transform(combined_df)

# Run PCA for 5 components
pca = PCA(n_components=5)
pca_result = pca.fit_transform(data_std)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained Variance Ratio:")
print(explained_variance_ratio)

#Draw a scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()


# In[131]:


#3C -- Based on the explained variance and the scree plot, I want to keep 3 of the principal components. In particular, I want to keep
#the top three principal components, because they are above the elbow of the scree plot, meaning they contain much more information about
#the data than the components below the elbow. 


# In[163]:


#3D -- Save the selected components to a new dataframe
pca_result_sliced = pca_result[:, :3]
data_pca=pd.DataFrame(pca_result_sliced,columns=['PC1','PC2','PC3'])
data_pca.head()


# In[ ]:




