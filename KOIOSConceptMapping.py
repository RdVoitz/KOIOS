#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

dataset = pd.read_excel('koiosDatabase_Concepts_and_Definitions.xlsx')


# In[4]:


dataset.head()


# In[3]:


for column in dataset.columns:
    print(column)


# In[4]:


list(dataset)   


# In[5]:


needed_Attributes = ['CONCEPT_ID','TERM_ID','DEFINITION_ID','SYNONYMS_ID','SYNONYM_VALUE','DEFINITION','CONCEPT_TYPE_ID',
'DEFINITION_CONTENT_ID','TERM_SOURCE_ID','DEFINITION_SOURCE_ID']

for x in dataset:
    if x not in needed_Attributes:
        dataset.drop(x, 1, inplace = True)


# In[6]:


list(dataset)


# In[7]:


print(len(list(dataset)))


# In[8]:


print(len(dataset))


# In[9]:


dataset[dataset['SYNONYM_VALUE']=='standard form']['DEFINITION'] 


# In[10]:


print(len(dataset[dataset['SYNONYM_VALUE']=='standard form']['DEFINITION']))


# In[11]:


print(dataset['SYNONYM_VALUE'])


# In[12]:


all_values = []
duplicates = []
count = 0

for x in dataset['SYNONYM_VALUE']:
    if x in all_values and x not in duplicates:
        count = count + 1
        duplicates.append(x)
    else:
        all_values.append(x)
print(count)


# In[13]:


all_values = []
duplicates = []
count = 0

for x in dataset['CONCEPT_ID']:
    if x in all_values and x not in duplicates:
        count = count + 1
        duplicates.append(x)
    else:
        all_values.append(x)
print(count)


# In[14]:


# Term - Concept Dictionary/Mapping : To determine similar concepts i.e. concepts and corresponding definitions containing same synonym values.
term_concept = dict.fromkeys(set(dataset['SYNONYM_VALUE']))

for index, row in dataset.iterrows():
    term = row['SYNONYM_VALUE']
    concept_id = row['CONCEPT_ID']
    concept_type = row['CONCEPT_TYPE_ID']
    definition = row['DEFINITION']
    
    if term_concept[term] == None:
        term_concept[term] = [(concept_id, concept_type, definition)]
    else:
        term_concept[term] = term_concept[term] + [(concept_id, concept_type, definition)]


# In[15]:


print(len(term_concept['door']))


# In[16]:


for key in list(term_concept.keys()):
    if len(term_concept[key]) == 1:
        del term_concept[key]
print(len(term_concept))


# In[17]:


print(len(term_concept['wetting agent']))


# In[7]:


# Case 1 - Text similarity with Gensim on Sample Data

import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.matutils import softcossim
from gensim import corpora


# In[8]:


import gensim.downloader as api

# fast text model - pretrained models through the downloader api (embedder model)
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')


# In[20]:


sentence_1 = remove_stopwords('Sachin is a cricket player and an opening batsman').split()
sentence_2 = remove_stopwords('Dhoni is a cricket player too He is a batsman and keeper').split()
sentence_3 = remove_stopwords('Anand is a chess player').split()
sentence_4 = remove_stopwords('This is such a sunny day').split()

print(sentence_1)
print(sentence_2)
print(sentence_3)
print(sentence_4)

# Dictionary and Corpus
documents = [sentence_1, sentence_2, sentence_3, sentence_4]
dictionary = corpora.Dictionary(documents)

# Composing the similarity matrix
similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)

# Convert sentences into bag-of-words vectors.
sentence_1 = dictionary.doc2bow(sentence_1)
sentence_2 = dictionary.doc2bow(sentence_2)
sentence_3 = dictionary.doc2bow(sentence_3)
sentence_4 = dictionary.doc2bow(sentence_4)

print(sentence_1)
print(sentence_3)
print(sentence_3)
print(sentence_4)

# Soft cosine similarity
print(softcossim(sentence_1, sentence_2, similarity_matrix))
print(softcossim(sentence_1, sentence_3, similarity_matrix))
print(softcossim(sentence_2, sentence_3, similarity_matrix))
print(softcossim(sentence_2, sentence_4, similarity_matrix))


# In[21]:


print(dataset['SYNONYM_VALUE'])


# In[86]:


# Testing Gensim with the actual KOIOS data
# Step 1 - Clean data (Removing stopwords and punctuation)

from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.matutils import softcossim
from gensim import corpora
from gensim.models import TfidfModel

def clean_compute_similarity(d1,d2):
    
    #print(type(d1))
    #print(type(d2))
    
    d1 = remove_stopwords(d1).split()
    d2 = remove_stopwords(d2).split()
    
    #print(d1)
    #print(d2)
    
    # Dictionary and Corpus
    documents = [d1, d2]
    dictionary = corpora.Dictionary(documents)
    
    # Composing the similarity matrix
    similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)
    
    # Conversion of sentences into bag-of-words vectors - The function doc2bow() simply counts the number of occurrences of each distinct word, converts the word to its integer word id and returns the result as a sparse vector.
    d1 = dictionary.doc2bow(d1)
    d2 = dictionary.doc2bow(d2)
    
    #print(d1)
    #print(d2)
    
    # Soft cosine similarity - Considers similarities between pairs of features
    score = softcossim(d1, d2, similarity_matrix)
    
    return score


# In[87]:


# Term - Concept Dictionary/Mapping : To determine similar concepts i.e. concepts and corresponding definitions containing same synonym values.
term_concept = dict.fromkeys(set(dataset['SYNONYM_VALUE']))

for index, row in dataset.iterrows():
    term = row['SYNONYM_VALUE']
    concept_id = row['CONCEPT_ID']
    concept_type = row['CONCEPT_TYPE_ID']
    definition = row['DEFINITION']
    
    if term_concept[term] == None:
        term_concept[term] = {}
    
    # Dictionary mapping from terms - concept_id | concept_id - concept_type, definition
    term_concept[term][concept_id] = (concept_type, definition)


# In[89]:


tc_keys = list(term_concept.keys())
#print(tc_keys)
similar_cd = []
similar_type = dict.fromkeys(set(tc_keys))

for term in tc_keys:
    cd_keys = list(term_concept[term].keys())
    for x in range(0, len(cd_keys)-1):
        concept_id1 = cd_keys[x]
        type_id1, description1 = term_concept[term][concept_id1]
        for y in range(x+1, len(cd_keys)):
            concept_id2 = cd_keys[y]
            type_id2, description2 = term_concept[term][concept_id2]
            
            if type_id1 == type_id2:
                if clean_compute_similarity(description1, description2) >= 0.99:
                    similar_cd.append((description1, description2))
                    if similar_type[term] == None:
                        similar_type[term] = []
                    else:
                        similar_type[term] = similar_type[term] + [(description1, description2)]
            


# In[90]:


print(len(similar_cd))


# In[91]:


print(len(similar_type))


# In[92]:


print(len(similar_type.keys()))


# In[93]:


print(similar_cd)


# In[94]:


def most_Similar():
    check = 0
    count = 0
    description = {}
    memorised = None
    
    for key in tc_keys:
        if not similar_type[key] == None:
            count = count+1
            if(len(similar_type[key]) > check):
                memorised = key
                check = len(similar_type[key])
                
    print(check)
    print(memorised)
    print(count)
    print('**************************')
    
    for tuple in similar_type[memorised]:
        d1, d2 = tuple
        if d1 not in list(description.keys()):
            description[d1] = [d1,d2]
        else:
            description[d1] = description[d1] + [d2]
            
    check = 0
    memorised = None
    for d in list(description.keys()):
        if len(set(description[d])) > check:
            check = len(set(description[d]))
            memorised = d
        
               
    print(len(set(description[memorised])))
    print(memorised)
    print('**************************')
    
    for d in set(description[memorised]):
        print(d)
        
most_Similar()        


# In[1]:


# Text preprocessing with Spacy
import spacy

nlp = spacy.load('en_core_web_md')


# In[ ]:




