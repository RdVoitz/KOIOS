#!/usr/bin/env python
# coding: utf-8

# In[24]:


import requests
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_lg')


# In[105]:


#Utility URLs for accessing TARIFF API

url = "https://www.trade-tariff.service.gov.uk"
section_nr = 16

section_url = 'https://www.trade-tariff.service.gov.uk/v2/sections/'
heading_url = 'https://www.trade-tariff.service.gov.uk/v1/headings/'
response = requests.get(section_url + str(section_nr) + '.json')


# In[107]:


#RETREIVE HC2 and HC4 lists after retreiving data from API
#HC2 contains the boundaries of the HC2 code at the current section
#HC4 contains tuples of HC4 codes representing bottom and upper boundaries

HC2 = []
HC4 = []

data = response.json()
HC2.append(data['data']['attributes']['chapter_from'])
HC2.append(data['data']['attributes']['chapter_to'])

for x in data['included']:
    if 'attributes' in set(x.keys()):
        if 'headings_from' and 'headings_to' and 'description' in set(x['attributes'].keys()):
            HC4.append((x['attributes']['headings_from'], x['attributes']['headings_to'])) #, x['attributes']['description']


# In[108]:


#testing if HC2 and HC4 contain correct information

print('HC2 CODES: ', HC2)
for tup in HC4:
    fro, to = tup
    print('HC4 BRANCH FROM', fro, 'TO ', to) #, ' WITH THE HC2 DESCRIPTION: ', d)


# In[28]:


#get data at HC4 level
def getData(code):
    response = requests.get(heading_url + str(code) + '.json')
    data = response.json()
    return data


# In[80]:


#compute cosine similarity between input description and HC4/HC6 descriptions
def HC4_mapping(inp,desc):
    desc = nlp(desc)
    score = inp.similarity(desc)
    
    return score

#create dictionary of HC4 codes pointing to similarity scores and HC4 level descriptions
def HC4_dictionary(inp, HC4):
    inp = nlp(inp)
    HC4_score = {}
    HC4_desc = {}

    for x in HC4:
        fro, to, d = x
        fro = int(fro)
        to = int(to)

        for i in range(fro, to+1):
            data = getData(i)
            
            if 'description' in set(data.keys()):
                #print("HERE")
                desc = data['description']
                similarity_score = HC4_mapping(inp, desc)
                HC4_score[i] = similarity_score
                HC4_desc[i] = desc
                
# PROVE THEM WRONG!                
#             else:
#                 print(data)

                
            similarity_score = HC4_mapping(inp, desc)
            HC4_score[i] = similarity_score
            
    return HC4_score, HC4_desc

#Sort the HC4 dictionary in descending order
#OR:  use a threshold
def best_HC4_codes(dictionary, threshold):
    
    best_HC4 = []
    sorted_x = sorted(dictionary.items(), reverse = True, key=lambda kv: kv[1])
    
#     for x in dictionary:
#         if dictionary[x] >= threshold:
#             best_HC4.append(x)
    
    return sorted_x
        


# In[81]:


#TESTING HC4 level similarity queries

inp = 'Fenner couplings'

HC4_dict, HC4_desc = HC4_dictionary(inp, HC4)
HC4_best = best_HC4_codes(HC4_dict, 0.2)

for x in HC4_best:
    code, score = x
    print("CODE: ", code, ", Similarity score: ", HC4_dict[code], ", Description: ", HC4_desc[code])
# for code in HC4_best:
#     print("CODE: ", code, ", Similarity score: ", HC4_dict[code], ", Description: ", HC4_desc[code])


# In[93]:


#RETREIVE HC6 level description using a given HC4 code and heading_url
def HC6_descriptions(HC4_code):
    
    HC6_desc = []
    
    request = requests.get(heading_url + str(HC4_code) + '.json')
    data = request.json()
    
    if 'commodities' in set(data.keys()):
        for x in data['commodities']:
            nr = x['number_indents']
            tid = x['goods_nomenclature_item_id']
            description = x['description']
            leaf = x['leaf']

            if nr == 1:
                HC6_desc.append((tid, description))
    return HC6_desc

#similar to HC4 but using HC6 descriptions as well to improve the result
def HC6_dictionary(inp, HC4):
    inp = nlp(inp)
    
    HC6_score = {}
    HC6_desc = {} #dictionary

    for x in HC4:
        fro, to, d = x
        fro = int(fro)
        to = int(to)

        for i in range(fro, to+1):
            data = getData(i)
            
            #DICTIONARY of HC4 codes containing dictionaries of HC6 codes
            #EACH HC6 code is pointing to a combined description(HC4 & HC6) and a similarity score (using the input)
            if i not in set(HC6_score.keys()): #HC6_score[i] == None:
                HC6_score[i] = {}
            if i not in set(HC6_desc.keys()):
                HC6_desc[i] = {}
            
            if 'description' in set(data.keys()):
                HC4_desc = data['description']
                HC6_d = HC6_descriptions(i)
                
                for description in HC6_d:
                    code, desc = description
                    modified_desc = HC4_desc + '; ' + desc
                    similarity_score = HC4_mapping(inp, modified_desc)
                    
                    HC6_score[i][code] = similarity_score
                    HC6_desc[i][code] = modified_desc
                    
    return HC6_score, HC6_desc

#retreive the best HC6 results using particular threshold
#MAY BE MODIFIED TO select TOP N results
def best_HC6_codes(dictionary, threshold):
    best_HC6 = []
    
    for hc4 in dictionary:
        for hc6 in dictionary[hc4]:
            if dictionary[hc4][hc6] >= threshold:
                best_HC6.append((hc4, hc6, dictionary[hc4][hc6]))
    
    return best_HC6


# In[109]:


#TESTING HC6 methods

inp = 'Fenner couplings'

HC6_score, HC6_desc = HC6_dictionary(inp, HC4)

HC6_best = best_HC6_codes(HC6_score, 0.22)

for x in HC6_best:
    hc4, hc6, score = x
    desc = HC6_desc[hc4][hc6]
    
    print("HC4 CODE: ", hc4)
    print("HC6 CODE: ", hc6)
    print("SCORE: ", score)
    print("Description: ", desc)
    print('****************')


# In[112]:


#RETREIVING INFORMATION AT HC6, HC8 and HC10 level


type(json['commodities'])

HC6 = []
HC8 = []
HC10 = []

for x in json['commodities']:
    nr = x['number_indents']
    tid = x['goods_nomenclature_item_id']
    description = x['description']
    leaf = x['leaf']
    
    if nr == 1:
        HC6.append((nr, tid, description, leaf))
    if nr == 2:
        HC8.append((nr, tid, description, leaf))
    if nr == 3:
        HC10.append((nr, tid, description, leaf))
    #print("*******************************")

for x in HC6:
    n,t,d,l = x
    
    print('Number indents: ',n)
    print('CODE: ',t)
    print('Description', d)
    print('Is Leaf?', l)
    print()
    print('*******************************')
    print()

for x in HC8:
    n,t,d,l = x
    
    print('Number indents: ',n)
    print('CODE: ',t)
    print('Description', d)
    print('Is Leaf?', l)
    print()
    print('*******************************')
    print()

for x in HC10:
    n,t,d,l = x
  
     if t == '8482101010':
        print(d)
        for token in nlp(d):
            print(token.text, token.pos_, token.left_edge)
        for ent in nlp(d).ents:
            print(ent.text, ent.label_)

        displacy.render(nlp(low), style="dep")
    
    print('Number indents: ',n)
    print('CODE: ',t)
    print(type(t))
    print('Description:', d)
    print('Is Leaf?', l)
    print()
    print('*******************************')
    print()

