#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd 
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# In[80]:

#file_path=os.path.join(os.getcwd(),(movies__final_data.csv))
#df=pd.read_csv(file_path)

df=pd.read_csv(r'D:\ENGINEERING\recommendation system\movies__final_data.csv')


# In[86]:


def cosine_sim(df):
    vectorizer=CountVectorizer(stop_words='english')
    count_matrix=vectorizer.fit_transform(df['document'])
    cos_sim=cosine_similarity(count_matrix,count_matrix)
    df=df.reset_index()
    indexs=pd.Series(df.index,df['title'])
    return indexs , cos_sim


# In[87]:


def get_recommendation (title):
    indexs,cos_sim=cosine_sim(df)
    idx=indexs[title]
    similarity_scores=list(enumerate(cos_sim[idx]))
    similarity_scores=sorted(similarity_scores,key=lambda x: x[1],reverse=True) #key with 
    #lambda function sorts according to the score itself as second element in each iteration in list
    similarity_scores=similarity_scores[1:11] #10 movies
    
    movies_idx=[ind[0] for ind in similarity_scores]
    movies=df['title'].iloc[movies_idx]
    #movies_list=[ind for ind in movies]

    return movies


# In[ ]:



# In[ ]:




