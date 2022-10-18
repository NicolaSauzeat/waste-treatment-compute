import multiprocessing
from multiprocessing import Pool
from gensim.models import Word2Vec
import torch
import seaborn
import pandas as pd
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW


class Compute:

    def __init__(self):
        self.pool = Pool(multiprocessing.cpu_count() - 1)
        self.path_in = "./Resultats/in/"
        self.path_out = "./Resultats/out/"

    def train_model(self, final_sentences):
        try:
            modele = int(input("""Choisissez un modèle parmis les suivants : /n 1: Word2Vec /n 2 : Word2Vec with lemma 
            /n  3 : Sentence2Vec /n 4 : All"""))
        except ValueError:
            modele = int(input(""""Votre saisie est incorrecte, ce doit être un nombre : /n 1: Word2Vec /n 2 : Word2Vec with lemma /n 
                  3 : Sentence2Vec /n 4 : All"""))
        if modele == 1:
            self.train_w2v_model(final_sentences["filtered_sentence"], model=1)
        elif modele == 2:
            self.train_w2v_model(final_sentences["lemma_sentence"], model=2)
        elif modele ==3:
            final_dataset = self.train_bert_model(final_sentences)
            self.train_s2v_model(final_dataset)

    def train_bert_model(self, final_sentences):
        validation_set = pd.read_csv(self.path_in +"jeu_annote_final.csv", sep="!", usecols=[2,3,4,5,6,7,8,9,10,11,12])
        validation_set_final = validation_set.merge(final_sentences, left_on="etablissement", right_on="name")


    def train_w2v_model(self, sentence, model):
        # Word2Vec model training
        model_cbow = Word2Vec(sentences=sentence, sg=0, min_count=5, workers=self.pool, window=3, epochs=150)
        # Save model
        if model == 1 :
            model_cbow.save("waste-treatment/sg_model_one.model")
        if model ==2 :
            model_cbow.save("waste-treatment/sg_model_two.model")

    def train_s2v_model(self, sentence):
        # Sentence2Vec model training
        #vectorizer.run(sentences, remove_stop_words = self.final_english_stopwords_list, add_stop_words = self.final_french_stopwords_list)
        # Save model
        #model_cbow.save("waste-treatment/sg_model_two.model")
        model_sent2vec = Sent2Vec(sentence["filtered_sentence"], size=100, min_count=1)
        #model_sent2vec.build_vocab(new_sentences, update=True)
        #model.train(new_sentences, total_examples=model.corpus_count, epochs=model.epochs)

    def train_d2v_model(self, sentence):
        # set up hyperparameters for building model
        model = Doc2Vec(workers=8, epochs=155)
        # build model vocabulary with corpus from sentences
        model.build_vocab(corpus_for_doc2vec)
        model.train(corpus_for_doc2vec, total_examples=model.corpus_count, epochs=model.epochs)
        model.save("d2vAllNAF_FINAL_DEUX.model")

    @staticmethod
    def choose_model(self, modele):
       if modele == '1':
            model = gensim.models.Word2Vec.load("waste-treatment/sg_model_one.model")
       elif model == "2":
            model = gensim.models.Word2Vec.load("waste-treatment/sg_model_two.model")
       elif model == "3":
            model = gensim.models.Word2Vec.load("waste-treatment/sg_model_two.model")
       #elif model == "4":
       #     model = gensim.models.


    def get_vector(self, sentence):
        sentence = self.clear_sentence(sentence)
        vectors = [self.model.wv[w] for w in word_tokenize(sentence, language='french')
                   if w in self.model.wv]

        weights = [1 / (math.log(self.word_weight.get(w, 10) + 0.00001)) for w in
                   word_tokenize(sentence, language='french')
                   if w in self.model.wv]
        v = np.zeros(self.model.vector_size)
        if (len(vectors) > 0):
            v = np.average(vectors, axis=0, weights=weights)  # Commande pour moyenne pondérée
            # v = (np.array([sum(x) for x in zip(*vectors)])) / v.size # Commande pour moyenne normale
        return v

    def tache_parallele(self, df, fonction):
        # create a pool for multiprocessing
        # split your dataframe to execute on these pools
        splitted_df = np.array_split(df, self.pool)
        # execute in parallel:
        split_df_results = self.pool.map(fonction, splitted_df)
        # combine your results
        df = pd.concat(split_df_results)
        pool.close()
        pool.join()
        return df


    def readWeights(self, frequency_file):
        data = open(frequency_file,"r").read()
        data = data.split('\n')
        del data[-1]
        self.word_weight = {}
        for d in data:
            wordpair = d.split(" ")
            wordpair[0] = wordpair[0][:-1]
            self.word_weight[wordpair[0]] = int(wordpair[1])

    def get_score(df, model):
        model.readWeights("statistiques.txt")
        for i, row in df.iterrows():
            if model.similarity(row["description"], row["WEB"]) >= 0.55:
                df.at[i, 'score'] = model.similarity(row["description"], row["WEB"])
            # scores.append(score)
            # cedcodes[idx] = sentence
            print(i)
            # df.at[i, 'score'] = score
        # df.to_csv('scores_juste2.csv', sep="!")
        return df

    def similarity(self, first_sentence, second_sentence):
        first_vector = self.get_vector(first_sentence)
        second_vector = self.get_vector(second_sentence)
        score = 0
        if first_vector.size > 0 and second_vector.size > 0:
            if (norm(first_vector) * norm(second_vector)) > 0:
                score = dot(first_vector, second_vector) / (norm(first_vector) * norm(second_vector)) # Similarité cosinus
        return score


"""
    def __init__(self,source_df, text_col, tag_col):
        self.source_df = source_df
        self.text_col = text_col
        self.tag_col = pd.read_csv("C:/Users/OpenStudio.Aurora-R13/Desktop/waste-treatment/Resultats/cpf2015_liste_n6.xlsx",
                                   header=1, usecols =["Code"])
        self.pool = Pool(multiprocessing.cpu_count() - 1)

    def __iter__(self):
        for i, row in self.source_df.iterrows():
            yield TaggedDocument(words=gensim.utils.simple_preprocess(row[self.text_col]),
                                 tags=[row[self.tag_col]])

"""