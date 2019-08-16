#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# In[39]:


data = pd.read_csv('E:/SHIKHA_FOLDER/My Machine learnig/Email-spam-detection/spam.csv',encoding='latin-1')


# In[40]:


data.head(10)


# In[41]:


data.drop(["Unnamed: 2"],axis=1,inplace=True)


# In[42]:


data.head(4)


# In[43]:


data.drop(["Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)


# In[44]:


data.head(4)


# In[45]:


data = data.rename(columns={"v1":"class", "v2":"text"})
data.head()


# In[46]:


data['length'] = data['text'].apply(len)
data.head()


# In[47]:


def pre_process(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words


# In[49]:


textFeatures = data['text'].copy()


# In[60]:


print(textFeatures)


# In[51]:


import nltk


# In[52]:


nltk.download('stopwords')


# In[61]:


textFeatures = textFeatures.apply(pre_process)


# In[62]:


print(textFeatures)


# In[54]:


vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(textFeatures)


# In[55]:


features_train, features_test, labels_train, labels_test = train_test_split(features, data['class'], test_size=0.3, random_state=111)


# In[56]:


from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# In[57]:


svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(features_train, labels_train)


# In[58]:


prediction = svc.predict(features_test)


# In[59]:


accuracy_score(labels_test,prediction)


# In[ ]:




