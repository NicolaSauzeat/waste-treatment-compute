from collections import Counter
from re import A
import re
import pandas as pd
import numpy as np
# Appel des bibliothèques
from numpy import dot
from numpy.linalg import norm
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import multiprocessing
from multiprocessing import Pool
import os
import spacy
import sent2vec
from sent2vec.vectorizer import Vectorizer
#vectorizer = Vectorizer()
from geotext import GeoText
from pathlib import Path

class Preprocessing:

    def __init__(self):
        self.final_french_stopwords_list = stopwords.words('french')
        self.final_english_stopwords_list = stopwords.words('english')
        self.path_in = Path("Resultats/in/")
        self.path_out = Path("Resultats/out/")
        self.pool = Pool(multiprocessing.cpu_count() - 1)

    def read_data(self):
        # Import file
        web_scrap = pd.read_csv(self.path_in /"BDD_avant_nettoyage.csv", sep=",")
        # Drop duplicates from dataframe
        web_scrap.drop_duplicates(subset="siren", inplace= True)
        web_scrap = web_scrap[(web_scrap["pages"] == web_scrap["pages"]) & (web_scrap["web"].str.len()>30)]
        # Drop empty rows
        web_scrap.dropna(subset=['web'], inplace=True)

        hs_code = pd.read_excel(self.path_in/"NC2017.xls", skiprows=2)
        hs_code_final = hs_code.loc[(hs_code["CHAPITRE 1"].str.len() >4) & (hs_code["CHAPITRE 1"].str.len() <8)]
        return web_scrap, hs_code_final

    def import_consolided_data(self):
        consolided_data = pd.read_csv(self.path_out/"consolided_data.csv", sep="!", engine="python")
        return consolided_data

    def tokenize_sentence(self, scrap, lemmatize= True):
        # Define regex
        regex = re.compile('[^A-zÀ-ú]+')
        # Replace unwanted symbols
        scrap["web"] = scrap["web"].apply(lambda x: regex.sub(' ', x))
        #
        scrap["web"] = scrap["web"].apply(lambda x: x.replace("  "," "))
        if lemmatize == True:
            scrap = self.lemmatize_sentence(scrap)
        # Get tokenize sentences
        tokenized_sentences = self.get_token(scrap)
        return tokenized_sentences

    @staticmethod
    def get_token(x):
        # Tokenize all sentences
        x["token"] = x["web"].apply(lambda x: word_tokenize(x, language="french"))
        return x

    def clear_sentence(self, raw_sentence, model=2):
        if model == 1:
            raw_sentence = self.get_lower(raw_sentence)
            raw_sentence = self.stop_word_removal(raw_sentence)
        else:
            # Extend to week days and months
            self.final_french_stopwords_list.extend(["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre",
                 "novembre", "decembre", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"])
            # Extend to website syntax
            self.final_french_stopwords_list.extend(["https", "www", "mail", "tel", "fax", "info", "français", "anglais", "mention", "légales", "linkedin",
                 "politique protection données", "contact", "twitter", "youtube", "facebook", "page", "accueil", "http","go","next"])
            places = []
            for row in raw_sentence["web"]:
                # Extend stop_word with cities names
                places.extend(GeoText(row).cities)
                # Extend stop_word with countries names
                places.extend(GeoText(row).countries)
                # Extend stop_word with countries names
                places.extend(GeoText(row).nationalities)
            self.final_french_stopwords_list.extend(places)
            raw_sentence = self.get_lower(raw_sentence)
            raw_sentence = self.stop_word_removal(raw_sentence)
        return raw_sentence

    @staticmethod
    def get_lower(raw_sentence):
        # Get all sentences in lower characters
        raw_sentence["filtered_sentence"] = raw_sentence["token"].apply(lambda x: [tok.lower() for tok in x])
        return raw_sentence

    def stop_word_removal(self, raw_sentence):
        # Stop Words removal in french
        raw_sentence["filtered_sentence"] = raw_sentence["filtered_sentence"].apply(lambda x: [tok for tok in x
                                    if len(tok) > 2 and (tok not in self.final_french_stopwords_list)])
        # Stop Words removal in english
        raw_sentence["filtered_sentence"] = raw_sentence["filtered_sentence"].apply(lambda x: [tok for tok in x
                                    if len(tok) > 2 and (tok not in self.final_english_stopwords_list)])
        return raw_sentence

    @staticmethod
    def lemmatize_sentence(raw_sentence):
        # Load lemmatize vocab for french language
        nlp = spacy.load('fr_core_news_sm')
        # Get all lemmatized format
        #raw_sentence["lemmatized_sentence"] = raw_sentence["filtered_sentence"].apply(lambda x: nlp(x))
        raw_sentence["lemmatize"] = ""
        nlp.max_length = 1030000000
        i = 0
        for doc in raw_sentence["web"]:
            i += 1
            if i == 1619 or i==2515 or i==2514:
                print('coucou')
            else:
                lemmatize = []
                try:
                    if len(doc) > 10000:
                        doc_1 = doc[slice(0, len(doc)//3)]
                        doc_2 = doc[slice(1,len(doc)//3)]
                        doc_3 = doc[slice(2, len(doc)//3)]
                        lemma_1 = nlp(doc_1)
                        lemma_2 = nlp(doc_2)
                        lemma_3 = nlp(doc_3)
                        lemmatize.extend([token.lemma_ for token in lemma_1 if token not in lemmatize])
                        lemmatize.extend([token.lemma_ for token in lemma_2])
                        lemmatize.extend([token.lemma_ for token in lemma_3])
                        raw_sentence.loc[raw_sentence["web"] == doc, "lemmatize"] = " ".join([token for token in lemmatize])
                    else:
                        lemma = nlp(doc)
                        lemmatize.extend([token.lemma_ for token in lemma if token not in lemmatize])
                        raw_sentence.loc[raw_sentence["web"] == doc, "lemmatize"] = " ".join([token for token in lemmatize])
                except ValueError:
                    print("oups")
        raw_sentence.rename(columns={"web": "RAW", "lemmatize": "web"}, inplace=True)
        return raw_sentence
