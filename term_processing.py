#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# def main():

# if __name__ == "__main__":
#     main()


# In[ ]:


#from CombinedVersion_TFIDF_TopicModelling import


# In[9]:


import nltk
from nltk.corpus import wordnet

syns = wordnet.synsets("x-ray_tube")
#syns
if syns == None:
    print('HELLO')
print(syns[0].lemmas()[0].name())
print(syns[0].definition()) 


# In[4]:


# Term-based matching methods
# Case 1: When given term is found in term-concept dictionary
#MAY INCLUDE Lemmatisation !!!!
def check_dictionary(term, term_concept, nlp, knn, n_neighbors, remove,
                     bi, dictionary, tfidf, lda, nr_topics, l1):
    keys = set(term_concept.keys())
    if term in keys:
        print("Match found")
        result = term_concept[term]
        
        return result
    else:
        print("Checking synonyms")
        check_synonym_dictionary(term, term_concept, nlp, knn, n_neighbors,
                                 remove, bi, dictionary, tfidf, lda, nr_topics,
                                 l1)
    


# In[1]:


def prepare_text(text,remove,bi,dictionary,tfidf,lda,nr_topics):
    
    text = bi[remove(text)]
    text_vec = dictionary.doc2bow(text)
    v = tfidf.transform([' '.join(text)]).toarray()

    #preprocess the new text
    processed = numpy.zeros(nr_topics)
    for tuples in lda[text_vec]:
        topic, score = tuples
        processed[topic] = score

    #new = [processed]                             
    new = [numpy.concatenate((v[0], processed), axis=0)]
    
    return new


# In[2]:


def return_top_n(text, knn, n_neighbors,l1):
    top = knn.kneighbors(text, n_neighbors)#, algorithm = 'auto')    
    x,pos = top
    print(top)
    for i in pos[0]:
        print(l1[i])
    #    print("l1 version of the sentence: ",l1[i])
        print("************************************************************************")


# In[3]:


# Case 2: When term not found, generate synonyms and search against term-concept dictionary
# Case 3: When synonym match not found, generate corresponding synonym definition; compute similarity and if score exceeds threshold, then print corresponding concept id, type and definition
def check_synonym_dictionary(synonym, term_concept, nlp, knn, n_neighbors
                             ,remove,bi,dictionary,tfidf,lda,nr_topics, l1):

    keyset = set(term_concept.keys())
    #keylist = list(term_concept.keys())
    syns = wordnet.synsets(synonym)
    
    if(len(syns)!= 0):
        for x in syns:
            
            syn_definition = nlp(x.definition())
            
            if x in keyset:
                print("Found", term_concept[x])
                return term_concept[x]
            else:
                print("Perform search against definitions of synonyms")
                
                
                text = prepare_text(syn_definition,remove, bi, dictionary,
                                   tfidf, lda, nr_topics)
                return_top_n(text, knn, n_neighbors, l1)
                
                
                for term in term_concept:
                    for concept in term_concept[term]: #list(term_to_concepts[term]):
                        types, definition = term_to_concepts[term][concept]
                        
                        des_definition = nlp(definition)
                        result = syn_definition.similarity(des_definition)

                        #print(result)
                        
                        if (result >= 0.85):
                            print("Success!")
                            check = term_concept[term][concept]
                            print("Synonym Definition: ", x.definition())
                            print("Results: ", concept, check)
                            print("Similarity Score", result)
                            return check
                        
                        else:
                            #print("Add new term into dictionary!")
                            break


# In[ ]:




