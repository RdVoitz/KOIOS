#!/usr/bin/env python
# coding: utf-8

# In[2]:


from socket import *
import requests


# In[1]:


import pandas as pd

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
        
print(df.columns.values)


# In[4]:


print(len(df.columns.values))


# In[5]:


# for index, row in df.iterrows():
#     term = row['SYNONYM_VALUE']
#     if term == 'standard form':
#         print(row['DEFINITION'],'  ', row['CONCEPT_TYPE_ID'], ' ', row['ECCMA_CONCEPT_ID'])


# In[6]:


keys = set(df['SYNONYM_VALUE'])
term_to_concepts = dict.fromkeys(keys)


for index, row in df.iterrows():
    term = row['SYNONYM_VALUE']
    concept_id = row['CONCEPT_ID']
    concept_type = row['CONCEPT_TYPE_ID']
    definition = row['DEFINITION']
    
    if term_to_concepts[term] == None:
        term_to_concepts[term] = {}
    
    term_to_concepts[term][concept_id] = (concept_type, definition)


# In[18]:


print(len(term_to_concepts['wetting agent']))
#for x in term_to_concepts['door']:
    


# In[7]:


def deletekeys():
    for x in list(term_to_concepts.keys()):
        if len(term_to_concepts[x]) == 1:
            del term_to_concepts[x]

deletekeys()
print(len(term_to_concepts))


# In[8]:


# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# stop_words = set(stopwords.words('english'))
# punctuation = [',','.','!','?',';',':','-']

# def clean(description):
#     d = description
#     word_tokens = word_tokenize(d)
#     fiktered = []
#     filtered = [w.lower() for w in word_tokens if not w in stop_words and not w in punctuation]
#     return filtered


# In[9]:


# example_sent = "this is a Sample sentence, showing off the Stop words filtration"
# print(clean(example_sent))


# In[ ]:


import spacy
nlp = spacy.load('en_core_web_lg')


# In[67]:


from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer

def compare(d1,d2):
    doc1 = nlp(d1)
    doc2 = nlp(d2)
    text = [d1,d2]
#     for t in doc1:
#         if not t.is_stop | t.is_punct:
#             print(t.vector)
#             break
    
#     doc1 = nlp(' '.join([str(t) for t in doc1 if not t.is_stop | t.is_punct ]))
#     doc2 = nlp(' '.join([str(t) for t in doc2 if not t.is_stop | t.is_punct ]))
    
    doc1 = ' '.join([str(t) for t in doc1 if not t.is_stop | t.is_punct ])
    doc2 = ' '.join([str(t) for t in doc2 if not t.is_stop | t.is_punct ])
    
    text = [doc1, doc2]
    
    
    tfidf = TfidfVectorizer()
    tfidf.fit(text)
    v1 = tfidf.transform([text[0]]).toarray()
    v2 = tfidf.transform([text[1]]).toarray()
    
   # tfidf.fit()
#     print(vector.shape)
#    print(vector.toarray())
#     print("************************")
            
#     v1 = doc1.vector
#     v2 = doc2.vector
    result = 1 - spatial.distance.cosine(v1,v2)
    #print(result)
    #score = doc1.similarity(doc2)  
    return result


# In[68]:


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


# In[ ]:





# In[73]:


doublekeys = list(term_to_concepts.keys())
similar = []
type_to_similar = dict.fromkeys(set(doublekeys))

for term in doublekeys:
    smalkeys = list(term_to_concepts[term].keys())
    for i in range(0, len(smalkeys)-1):
        conceptid1 = smalkeys[i]
        typeid1, description1 = term_to_concepts[term][conceptid1]
        
        for j in range(i+1, len(smalkeys)):
            conceptid2 = smalkeys[j]
            typeid2, description2 = term_to_concepts[term][conceptid2]
            
            if typeid1 == typeid2:
                if compare(description1,description2) >= 0.90:
                    similar.append((description1,description2))
                    if type_to_similar[term] == None:
                        type_to_similar[term] = []
                    type_to_similar[term] = type_to_similar[term] + [(description1,description2)]
                    
                    
        


# In[74]:


print(len(similar))

print(len(type_to_similar))

print(len(type_to_similar.keys()))


# In[75]:


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
    print(m)
    print(memorised)
    print(count)
    print('**************************')
    
    memorised = "standard form"
    
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




