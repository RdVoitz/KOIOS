#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

dataset = pd.read_excel('koiosDatabase_Concepts_and_Definitions.xlsx')


# In[2]:


dataset.head()


# In[3]:


for column in dataset.columns:
    print(column)


# In[78]:


list(dataset)   


# In[26]:


needed_Attributes = ['CONCEPT_ID','TERM_ID','DEFINITION_ID','SYNONYMS_ID','SYNONYM_VALUE','DEFINITION','CONCEPT_TYPE_ID',
'DEFINITION_CONTENT_ID','TERM_SOURCE_ID','DEFINITION_SOURCE_ID']

for x in dataset:
    if x not in needed_Attributes:
        dataset.drop(x, 1, inplace = True)


# In[27]:


list(dataset)


# In[22]:


print(len(list(dataset)))


# In[19]:


print(len(dataset))


# In[76]:


dataset[dataset['SYNONYM_VALUE']=='standard form']['DEFINITION'] 


# In[75]:


print(len(df[df['SYNONYM_VALUE']=='standard form']['DEFINITION']))


# In[29]:


print(dataset['SYNONYM_VALUE'])


# In[33]:


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


# In[34]:


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


# In[59]:


# Term - Concept Dictionary/Mapping : To determine similar concepts i.e. concepts and corresponding definitions containing same synonym values.
term_concept = dict.fromkeys(set(dataset['SYNONYM_VALUE']))

for index, row in dataset.iterrows():
    term = row['SYNONYM_VALUE']
    concept_id = row['CONCEPT_ID']
    concept_type = row['CONCEPT_TYPE_ID']
    definition = row['DEFINITION']
    
    if term_concept[term] == None:
        term_concept[term] = [(concept, concept_type, definition)]
    else:
        term_concept[term] = term_concept[term] + [(concept, concept_type, definition)]


# In[60]:


print(len(term_concept['door']))


# In[62]:


for key in list(term_concept.keys()):
    if len(term_concept[key]) == 1:
        del term_concept[key]
print(len(term_concept))


# In[64]:


print(len(term_concept['wetting agent']))


# In[3]:


# Case 1 - Text similarity with Gensim on Sample Data

from gensim.parsing.preprocessing import remove_stopwords
from gensim.matutils import softcossim
from gensim import corpora


# In[1]:


import gensim.downloader as api

# fast text model - pretrained models through the downloader api (embedder model)
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')


# In[4]:


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


# In[ ]:




