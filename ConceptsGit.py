{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from socket import *\n",
    "import requests\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "df = pd.read_excel ('koiosDatabase_Concepts_and_Definitions.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'\n",
    "#print(df.columns.values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    }
   ],
   "source": [
    "values = []\n",
    "repeated = []\n",
    "count = 0\n",
    "for x in df['CONCEPT_ID']:\n",
    "    if x in values and x not in repeated:\n",
    "        count = count + 1\n",
    "        repeated.append(x)\n",
    "    else:\n",
    "        values.append(x)\n",
    "print(count)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['CONCEPT_ID' 'TERM_ID' 'DEFINITION_ID' 'DEFINITION_CONTENT_ID'\n",
      " 'SYNONYMS_ID' 'CONCEPT_TYPE_ID' 'SYNONYM_VALUE' 'DEFINITION'\n",
      " 'TERM_SOURCE_ID' 'DEFINITION_SOURCE_ID' 'DEF_FULL_SOURCE_TEXT'\n",
      " 'ECCMA_CONCEPT_ID']\n"
     ]
    }
   ],
   "source": [
    "usefulAttributes = ['CONCEPT_ID', 'TERM_ID', 'DEFINITION_ID' ,'DEFINITION_CONTENT_ID','SYNONYMS_ID', \n",
    "'CONCEPT_TYPE_ID', 'SYNONYM_VALUE', 'DEFINITION','DEF_FULL_SOURCE_TEXT','TERM_SOURCE_ID', 'DEFINITION_SOURCE_ID', 'ECCMA_CONCEPT_ID']\n",
    "\n",
    "for x in df.columns.values:\n",
    "    if x not in usefulAttributes:\n",
    "        df.drop(x, 1, inplace=True)\n",
    "        \n",
    "print(df.columns.values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "12\n"
     ]
    }
   ],
   "source": [
    "print(len(df.columns.values))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A set of quad small package outline styles with gull-wing leads in standard form of which each outline style can be described with the same group of data element types    1   0161-1#01-067098#1\n",
      "A set of leadless rectangular package outline styles in standard form with end connections of which each outline style can be described with the same group of data element types    1   0161-1#01-067094#1\n",
      "A set of rectangular package outline styles with wrap-around end connections in standard form of which each outline style can be described with the same group of data element types    1   0161-1#01-067095#1\n",
      "A set of triple-in-line package outline styles with through-hole leads in standard form of which each outline style can be described with the same group of data element types    1   0161-1#01-067081#1\n",
      "A set of zigzag-in-line package outline styles with through-hole leads in standard form of which each outline style can be described with the same group of data element types    1   0161-1#01-067082#1\n",
      "A set of dual-in-line package outline styles with through-hole leads in standard form of which each outline style can be described with the same group of data element types    1   0161-1#01-067078#1\n",
      "A set of quad-in-line package outline styles with through-hole leads in standard form of which each outline style can be described with the same group of data element types    1   0161-1#01-067079#1\n",
      "A set of single-in-line package outline styles with through-hole leads in standard form of which each outline style can be described with the same group of data element types    1   0161-1#01-067080#1\n",
      "A set of quad flat-pack package outline styles with gull-wing leads in standard form of which each outline style can be described with the same group of data element types    1   0161-1#01-067073#1\n",
      "A set of cylindrical package outline styles with upper screw connections in standard form of which each outline style can be described with the same group of data element types    1   0161-1#01-066902#1\n",
      "A set of cylindrical package outline styles with wrap-around end connections in standard form of which each outline style can be described with the same group of data element types    1   0161-1#01-066898#1\n",
      "A set of cylindrical package outline styles with upper solder-lug connections in standard form of which each outline style can be described with the same group of data element types    1   0161-1#01-066901#1\n",
      "A set of quad chip-carrier package outline styles with standard J-bend leads of which each outline style can be described with the same group of data element types    1   0161-1#01-066890#1\n",
      "A set of quad leadless chip-carrier package outline styles with standard leads of which each outline style can be described with the same group of data element types    1   0161-1#01-066891#1\n"
     ]
    }
   ],
   "source": [
    "for index, row in df.iterrows():\n",
    "    term = row['SYNONYM_VALUE']\n",
    "    if term == 'standard form':\n",
    "        print(row['DEFINITION'],'  ', row['CONCEPT_TYPE_ID'], ' ', row['ECCMA_CONCEPT_ID'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "keys = set(df['SYNONYM_VALUE'])\n",
    "term_to_concepts = dict.fromkeys(keys)\n",
    "\n",
    "\n",
    "for index, row in df.iterrows():\n",
    "    term = row['SYNONYM_VALUE']\n",
    "    concept_id = row['CONCEPT_ID']\n",
    "    concept_type = row['CONCEPT_TYPE_ID']\n",
    "    definition = row['DEFINITION']\n",
    "    \n",
    "    if term_to_concepts[term] == None:\n",
    "        term_to_concepts[term] = {}\n",
    "    \n",
    "    term_to_concepts[term][concept_id] = (concept_type, definition)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "7\n"
     ]
    }
   ],
   "source": [
    "print(len(term_to_concepts['wetting agent']))\n",
    "#for x in term_to_concepts['door']:\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5084\n"
     ]
    }
   ],
   "source": [
    "def deletekeys():\n",
    "    for x in list(term_to_concepts.keys()):\n",
    "        if len(term_to_concepts[x]) == 1:\n",
    "            del term_to_concepts[x]\n",
    "\n",
    "deletekeys()\n",
    "print(len(term_to_concepts))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\radu_\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "# import nltk\n",
    "# nltk.download('stopwords')\n",
    "# from nltk.corpus import stopwords\n",
    "# from nltk.tokenize import word_tokenize\n",
    "\n",
    "# stop_words = set(stopwords.words('english'))\n",
    "# punctuation = [',','.','!','?',';',':','-']\n",
    "\n",
    "# def clean(description):\n",
    "#     d = description\n",
    "#     word_tokens = word_tokenize(d)\n",
    "#     fiktered = []\n",
    "#     filtered = [w.lower() for w in word_tokens if not w in stop_words and not w in punctuation]\n",
    "#     return filtered"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['sample', 'sentence', 'showing', 'stop', 'words', 'filtration']\n"
     ]
    }
   ],
   "source": [
    "# example_sent = \"this is a Sample sentence, showing off the Stop words filtration\"\n",
    "# print(clean(example_sent))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy\n",
    "nlp = spacy.load('en_core_web_lg')\n",
    "\n",
    "def compare(d1,d2):\n",
    "    doc1 = nlp(d1)\n",
    "    doc2 = nlp(d2)\n",
    "    \n",
    "    doc1 = nlp(' '.join([str(t) for t in doc1 if not t.is_stop | t.is_punct ]))\n",
    "    doc2 = nlp(' '.join([str(t) for t in doc2 if not t.is_stop | t.is_punct ]))\n",
    "    score = doc1.similarity(doc2)  \n",
    "    return score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n",
      "C:\\Users\\radu_\\Anaconda3\\lib\\runpy.py:193: UserWarning: [W008] Evaluating Doc.similarity based on empty vectors.\n",
      "  \"__main__\", mod_spec)\n"
     ]
    }
   ],
   "source": [
    "doublekeys = list(term_to_concepts.keys())\n",
    "similar = []\n",
    "type_to_similar = dict.fromkeys(set(doublekeys))\n",
    "\n",
    "for term in doublekeys:\n",
    "    smalkeys = list(term_to_concepts[term].keys())\n",
    "    for i in range(0, len(smalkeys)-1):\n",
    "        conceptid1 = smalkeys[i]\n",
    "        typeid1, description1 = term_to_concepts[term][conceptid1]\n",
    "        \n",
    "        for j in range(i+1, len(smalkeys)):\n",
    "            conceptid2 = smalkeys[j]\n",
    "            typeid2, description2 = term_to_concepts[term][conceptid2]\n",
    "            \n",
    "            if typeid1 == typeid2:\n",
    "                if compare(description1,description2) >= 0.95:\n",
    "                    similar.append((description1,description2))\n",
    "                    if type_to_similar[term] == None:\n",
    "                        type_to_similar[term] = []\n",
    "                    type_to_similar[term] = type_to_similar[term] + [(description1,description2)]\n",
    "                    \n",
    "                    \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2275\n",
      "5084\n",
      "5084\n"
     ]
    }
   ],
   "source": [
    "print(len(similar))\n",
    "\n",
    "print(len(type_to_similar))\n",
    "\n",
    "print(len(type_to_similar.keys()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "86\n",
      "standard form\n",
      "1499\n",
      "**************************\n",
      "13\n",
      "A set of quad small package outline styles with gull-wing leads in standard form of which each outline style can be described with the same group of data element types\n",
      "**************************\n",
      "A set of leadless rectangular package outline styles in standard form with end connections of which each outline style can be described with the same group of data element types\n",
      "A set of dual-in-line package outline styles with through-hole leads in standard form of which each outline style can be described with the same group of data element types\n",
      "A set of cylindrical package outline styles with wrap-around end connections in standard form of which each outline style can be described with the same group of data element types\n",
      "A set of zigzag-in-line package outline styles with through-hole leads in standard form of which each outline style can be described with the same group of data element types\n",
      "A set of single-in-line package outline styles with through-hole leads in standard form of which each outline style can be described with the same group of data element types\n",
      "A set of rectangular package outline styles with wrap-around end connections in standard form of which each outline style can be described with the same group of data element types\n",
      "A set of triple-in-line package outline styles with through-hole leads in standard form of which each outline style can be described with the same group of data element types\n",
      "A set of quad flat-pack package outline styles with gull-wing leads in standard form of which each outline style can be described with the same group of data element types\n",
      "A set of cylindrical package outline styles with upper screw connections in standard form of which each outline style can be described with the same group of data element types\n",
      "A set of quad chip-carrier package outline styles with standard J-bend leads of which each outline style can be described with the same group of data element types\n",
      "A set of quad leadless chip-carrier package outline styles with standard leads of which each outline style can be described with the same group of data element types\n",
      "A set of quad-in-line package outline styles with through-hole leads in standard form of which each outline style can be described with the same group of data element types\n",
      "A set of quad small package outline styles with gull-wing leads in standard form of which each outline style can be described with the same group of data element types\n"
     ]
    }
   ],
   "source": [
    "def mostSimilar():\n",
    "    m = 0\n",
    "    memorised = None\n",
    "    descriptions = {}\n",
    "    count = 0\n",
    "    \n",
    "    for term in set(doublekeys):\n",
    "        if not type_to_similar[term] == None: \n",
    "            count = count +1\n",
    "            if len(type_to_similar[term]) > m:\n",
    "                memorised = term\n",
    "                m = len(type_to_similar[term])\n",
    "    print(m)\n",
    "    print(memorised)\n",
    "    print(count)\n",
    "    print('**************************')\n",
    "    \n",
    "    for touple in type_to_similar[memorised]:\n",
    "        d1,d2 = touple\n",
    "        \n",
    "        if not d1 in list(descriptions.keys()):\n",
    "            descriptions[d1] = [d1,d2]\n",
    "        else:\n",
    "            descriptions[d1] = descriptions[d1] + [d2]\n",
    "    \n",
    "    m = 0\n",
    "    memorised = None\n",
    "    for d in list(descriptions.keys()):\n",
    "        if len(set(descriptions[d])) > m:\n",
    "            m = len(set(descriptions[d]))\n",
    "            memorised = d\n",
    "    \n",
    "    print(len(set(descriptions[memorised])))\n",
    "    print(memorised)\n",
    "    print('**************************')\n",
    "    \n",
    "    for d in set(descriptions[memorised]):\n",
    "        print(d)\n",
    "        \n",
    "mostSimilar()    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for index, row in df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "E1 & E2  0.9183546615204163\n",
      "E1 & E3  0.5687512811294366\n",
      "E1 & E4  0.2609567676624732\n",
      "E2 & E3  0.5383482092891331\n",
      "E2 & E4  0.193027597744168\n",
      "E3 & E4  0.1382400979348183\n"
     ]
    }
   ],
   "source": [
    "example1 = 'Sachin is a cricket player and a opening batsman'\n",
    "example2 = 'Dhoni is a cricket player too He is a batsman and keeper'\n",
    "example3 = 'Anand is a chess player'\n",
    "example4 = 'This is such a sunny day'\n",
    "\n",
    "print(\"E1 & E2 \", compare(example1,example2))\n",
    "print(\"E1 & E3 \", compare(example1,example3))\n",
    "print(\"E1 & E4 \", compare(example1,example4))\n",
    "print(\"E2 & E3 \", compare(example2,example3))\n",
    "print(\"E2 & E4 \", compare(example2,example4))\n",
    "print(\"E3 & E4 \", compare(example3,example4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
