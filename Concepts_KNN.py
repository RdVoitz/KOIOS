#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import spacy
import datetime

def read_trim():
    df = pd.read_excel ('koiosDatabase_Concepts_and_Definitions.xlsx')
    usefulAttributes = ['CONCEPT_ID', 'TERM_ID', 'DEFINITION_ID' ,'DEFINITION_CONTENT_ID','SYNONYMS_ID', 
    'CONCEPT_TYPE_ID', 'SYNONYM_VALUE', 'DEFINITION','DEF_FULL_SOURCE_TEXT','TERM_SOURCE_ID']
    for x in df.columns.values:
        if x not in usefulAttributes:
            df.drop(x, 1, inplace=True)  
    return df


# In[2]:


#create a dictionary of terms
def term_to_concepts(df):
    
    keys = set(df['SYNONYM_VALUE'])
    term_to_concepts = dict.fromkeys(keys)

    count = 0
    for index, row in df.iterrows():
        if count >-1 :
            count = count +1
            term = row['SYNONYM_VALUE']
            concept_id = row['CONCEPT_ID']
            concept_type = row['CONCEPT_TYPE_ID']
            definition = row['DEFINITION']

            if term_to_concepts[term] == None:
                term_to_concepts[term] = {}

            #use concept id as keys for the dictionary of each term
            term_to_concepts[term][concept_id] = (concept_type, definition)
        
    return term_to_concepts


# In[3]:


#delete terms with only one concept
def deletekeys(term_to_concepts):
    for x in list(term_to_concepts.keys()):
        if len(term_to_concepts[x]) == 1: #len(
            del term_to_concepts[x]


# In[18]:


#from sklearn.neighbors import KNeighborClassifier
from sklearn.neighbors import NearestNeighbors
import numpy as np

# method for training the tf-idf on all of the descriptions
def trainer():
    text = []
    for term in list(term_to_concepts.keys()):
        for concept in list(term_to_concepts[term]):
            t, d = term_to_concepts[term][concept]
            d = ' '.join([str(t) for t in nlp(d) if not t.is_stop | t.is_punct ])
            text.append(d)
    tfidf = TfidfVectorizer(ngram_range = (1,1), min_df = 10, max_df = 1000 )
    tfidf.fit(text)
    return tfidf

def trainer_w_term(term):
    text = []
    for concept in list(term_to_concepts[term]):
            t, d = term_to_concepts[term][concept]
            d = ' '.join([str(t) for t in nlp(d) if not t.is_stop | t.is_punct ])
            text.append(d)
    tfidf = TfidfVectorizer(ngram_range = (1,2))
    tfidf.fit(text)
    return tfidf 

def knn_trainer(tfidf):
    samples = []
    descriptions = []
    
    
    #vector_to_description = {}
    for term in list(term_to_concepts.keys()):
        for concept in list(term_to_concepts[term]):
            t, definition = term_to_concepts[term][concept]
            d = [' '.join([str(t) for t in nlp(definition) if not t.is_stop | t.is_punct ])]
            v = tfidf.transform(d).toarray()
            #print(v)
            #vector_to_description[v] = (concept, t, term, definition)
            samples.append(v[0])
            descriptions.append(definition)
    #print(samples)
    #print(len(samples))
    
    s = np.array(samples)
    print(s.shape)
    
    nn = NearestNeighbors(metric = 'cosine')
    nn.fit(s)
    
    return nn, descriptions

def return_top_k(k, d,nn, tfidf):
    
    d = ' '.join([str(t) for t in nlp(d) if not t.is_stop | t.is_punct ])
    v = tfidf.transform([d]).toarray()
    return nn.kneighbors([v[0]])
     


# In[5]:


#Compares two descriptions using the trained tfidf
#Returns similarity score measured with cosine
nlp = spacy.load('en_core_web_lg')
def compare(d1,d2, tfidf):
     
    #Clean the text
    doc1 = ' '.join([str(t) for t in nlp(d1) if not t.is_stop | t.is_punct ])
    doc2 = ' '.join([str(t) for t in nlp(d2) if not t.is_stop | t.is_punct ])
    
    text = [doc1, doc2]
    
    #get the word vector for each description to be compared
    v1 = tfidf.transform([text[0]]).toarray()
    v2 = tfidf.transform([text[1]]).toarray()
    
    result = 1 - spatial.distance.cosine(v1,v2)

    #result = doc1.similarity(doc2)  #Using spacy built in similarity
    return result


# In[7]:


df = read_trim()
term_to_concepts = term_to_concepts(df)
deletekeys(term_to_concepts)


# In[19]:


print("STARTED training TF-IDF AT: ",datetime.datetime.now())
tfidf = trainer()
print("FINISHED training TF-IDF AT: ",datetime.datetime.now())


# In[20]:


print("STARTED training KNN AT: ", datetime.datetime.now())
knn, description = knn_trainer(tfidf)
print("FINISHED training KNN AT: ",datetime.datetime.now())


# In[21]:


print("RETURN results for example: ", datetime.datetime.now())
test_sentence = "A set of triple-in-line package outline styles with through-hole leads in standard form of which each outline style can be described with the same group of data element types"
top = return_top_k(5,test_sentence, knn, tfidf)
print("Finished at: ", datetime.datetime.now())


# In[22]:


x,pos = top
print(top)
for i in pos[0]:
    print(description[i])


# In[ ]:




