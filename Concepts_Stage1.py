#!/usr/bin/env python
# coding: utf-8

# In[2]:


from socket import *
import requests


# In[1]:


import pandas as pd
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim

df = pd.read_excel ('koiosDatabase_Concepts_and_Definitions.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
#print(df.columns.values)


# In[2]:


values = []
repeated = []
count = 0
for x in df['CONCEPT_ID']:
    if x in values and x not in repeated:
        count = count + 1
        repeated.append(x)
    else:
        values.append(x)
print(count)


# In[3]:


usefulAttributes = ['CONCEPT_ID', 'TERM_ID', 'DEFINITION_ID' ,'DEFINITION_CONTENT_ID','SYNONYMS_ID', 
'CONCEPT_TYPE_ID', 'SYNONYM_VALUE', 'DEFINITION','DEF_FULL_SOURCE_TEXT','TERM_SOURCE_ID']

for x in df.columns.values:
    if x not in usefulAttributes:
        df.drop(x, 1, inplace=True)       
#print(df.columns.values)


# In[4]:


#create a dictionary of terms

keys = set(df['SYNONYM_VALUE'])
term_to_concepts = dict.fromkeys(keys)

for index, row in df.iterrows():
    term = row['SYNONYM_VALUE']
    concept_id = row['CONCEPT_ID']
    concept_type = row['CONCEPT_TYPE_ID']
    definition = row['DEFINITION']
    
    if term_to_concepts[term] == None:
        term_to_concepts[term] = {}
    
    #use concept id as keys for the dictionary of each term
    term_to_concepts[term][concept_id] = (concept_type, definition)
    


# In[5]:


#delete terms with only one concept
def deletekeys():
    for x in list(term_to_concepts.keys()):
        if len(term_to_concepts[x]) == 1:
            del term_to_concepts[x]

deletekeys()
print(len(term_to_concepts))


# In[6]:


import spacy
nlp = spacy.load('en_core_web_lg')


# In[7]:


#Compares two descriptions using the trained tfidf
#Returns similarity score measured with cosine

def compare(d1,d2, tfidf):
    doc1 = nlp(d1)
    doc2 = nlp(d2)
    
    #Clean the text
    doc1 = ' '.join([str(t) for t in doc1 if not t.is_stop | t.is_punct ])
    doc2 = ' '.join([str(t) for t in doc2 if not t.is_stop | t.is_punct ])
    
    text = [doc1, doc2]
    
    #get the word vector for each description to be compared
    v1 = tfidf.transform([text[0]]).toarray()
    v2 = tfidf.transform([text[1]]).toarray()
    
    result = 1 - spatial.distance.cosine(v1,v2)

    #result = doc1.similarity(doc2)  #Using spacy built in similarity
    return result


# In[8]:


# example1 = 'Sachin is a cricket player and a opening batsman'
# example2 = 'Dhoni is a cricket player too He is a batsman and keeper'
# example3 = 'Anand is a chess player'
# example4 = 'This is such a sunny day'

# print("E1 & E2 ", compare(example1,example2))
# print("E1 & E3 ", compare(example1,example3))
# print("E1 & E4 ", compare(example1,example4))
# print("E2 & E3 ", compare(example2,example3))
# print("E2 & E4 ", compare(example2,example4))
# print("E3 & E4 ", compare(example3,example4))


# In[9]:


# method for training the tf-idf on all of the descriptions
def trainer():
    text = []
    for term in list(term_to_concepts.keys()):
        for concept in list(term_to_concepts[term]):
            t, d = term_to_concepts[term][concept]
            d = ' '.join([str(t) for t in nlp(d) if not t.is_stop | t.is_punct ])
            text.append(d)
    tfidf = TfidfVectorizer()
    tfidf.fit(text)
    return tfidf
    


# In[12]:


#GENSIM METHOD

from gensim import corpora
from gensim.models import TfidfModel
import numpy as np
import pprint

def create_dictionary():
    text = []
    for term in list(term_to_concepts.keys()):
            for concept in list(term_to_concepts[term]):
                t, d = term_to_concepts[term][concept]
                d = [str(t).lower() for t in nlp(d) if not t.is_stop | t.is_punct ]
                text.append(d)
                #d = [d]
            break
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(doc, allow_update = True) for doc in text]

    tfidf = TfidfModel(corpus, smartirs='ntc')
    
    for doc in tfidf[corpus]:
        print([[dictionary[ids], np.around(freq, decimals=3)] for ids, freq in doc])
        
create_dictionary()


# In[72]:


import datetime
print(datetime.datetime.now())

#list of terms used as keys
doublekeys = list(term_to_concepts.keys()) #list of concepts

# list of touples made of similar descriptions
similar = [] 

#dictionary of
type_to_similar = dict.fromkeys(set(doublekeys))

#trained tfidf model
tfidf = trainer()  

for term in doublekeys:
    smalkeys = list(term_to_concepts[term].keys()) #list of concepts associated with a term
    
    #compare pairs of two descriptions by navigating the dictionary
    #of each term and addint the pairs with a level of similarity above a certain threshold
    #in a new dictionary associated with the term
    
    for i in range(0, len(smalkeys)-1):
        conceptid1 = smalkeys[i]
        typeid1, description1 = term_to_concepts[term][conceptid1]
        
        for j in range(i+1, len(smalkeys)):
            conceptid2 = smalkeys[j]
            typeid2, description2 = term_to_concepts[term][conceptid2]
            
            #the type of the concepts should be the same for comparison
            
            if typeid1 == typeid2:
                if compare(description1,description2,tfidf) >= 0.95:
                    similar.append((description1,description2)) #add pairs to a list just for keeping track 
                    if type_to_similar[term] == None:
                        type_to_similar[term] = []
                    type_to_similar[term] = type_to_similar[term] + [(description1,description2)]
                                       
print(datetime.datetime.now())        


# In[73]:


print(len(similar))

print(len(type_to_similar))

print(len(type_to_similar.keys()))


# In[76]:


def mostSimilar():
    m = 0
    memorised = None
    descriptions = {}
    count = 0
    
    for term in set(doublekeys):
        if not type_to_similar[term] == None: 
            count = count +1
            if len(type_to_similar[term]) > m:
                memorised = term
                m = len(type_to_similar[term])
    print("Maximum number of similar descriptions for a term  ",m)
    print("The term for which the max was found  ", memorised)
    print(count)
    print('**************************')
    
    memorised = "wire"
    
    for touple in type_to_similar[memorised]:
        d1,d2 = touple
        
        if not d1 in list(descriptions.keys()):
            descriptions[d1] = [d1,d2]
        else:
            descriptions[d1] = descriptions[d1] + [d2]
    
    m = 0
    memorised = None
    for d in list(descriptions.keys()):
        if len(set(descriptions[d])) > m:
            m = len(set(descriptions[d]))
            memorised = d
    
    print(len(set(descriptions[memorised])))
    print(memorised)
    print('**************************')
    
    for d in set(descriptions[memorised]):
        print(d)
        
mostSimilar()    


# In[ ]:


for index, row in df


# In[ ]:




