'''
En este archivo contiene la clase Scoring, que además realiza las representaciones de mensajes y palabras
basadas en scores.
'''

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
# import seaborn as sns
from operator import itemgetter
from math import exp, log
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#============ CLASE PARA CALCULAR EL SCORING ==================

class Scoring:

    def __init__(self,w2vmodel,W0:dict):
        self.W0 = W0
        self.model = w2vmodel
        self.vocab = list(w2vmodel.wv.key_to_index.keys())
        # Estas variables controlan el flujo y orden de los métodos:
        self.W0_check = False
        self.words_rep_check = False
        self.texts_rep_check = False
        self.scoring_check = False
        self.kv = False # Indica si el módelo son keyedvectors
    
    def build_neighbors(self,alpha=0):
        '''
        Esta función regresa un diccionario indexado con los vecinos más cercanos de la lista
        de palabras prototípicas W0_list. El valor del vecino es una tupla (max_sim,word_sim) donde
        max_sim: similitud con el vecino más cercano en W0_list
        word_sim: el vecino más cercano en W0_list
        '''
        model = self.model
        self.alpha = alpha
        W0_list = list(self.W0.keys())
        vecinos_W0 = {}
        N = len(self.vocab)
        filtered_W0 = [w for w in W0_list if w in self.vocab]
        for w in filtered_W0:
            vecinos_full = model.wv.similar_by_word(w,topn=N) 
            vecinos = []
            for pair in vecinos_full:
                if pair[1]>self.alpha:
                    vecinos.append(pair)
                else:
                    break
            # vecinos = [pair for pair in vecinos_full if pair[1]>self.alpha] 
            for pair in vecinos:
                if pair[0] not in W0_list:
                    if pair[0] in vecinos_W0.keys():
                        vecinos_W0[pair[0]].append((pair[1],w))
                    else:
                        vecinos_W0[pair[0]] = [(pair[1],w)]
        vecinos_W0 = {word:max(vecinos_W0[word],key=itemgetter(0)) for word in vecinos_W0.keys()}
        self.vecinos = vecinos_W0
        self.W0_check = True

    def __s_tilde(self,label:int,word:str):
        '''
        Esta función devuelve el pre-scoring de una palabra <word> en un mensaje con tag ds, la cual está normalizado entre -1 y 1.
        El diccionario <W0> contiene la lista de las palabras prototípicas como llaves y sus
        respectivos scores como values.
        El diccionario <W0_vecinos> contiene la lista de vecinos más cercanos a las palabras prototípicas 
        de W0 como llaves, los valores son tuplas (sim,vec) donde <vec> es el vecino más cercano
        en las palabras prototípicas y <sim> es la similitud que tiene con este vecino
        '''
        beta1,beta2 = self.beta1,self.beta2
        W0 = self.W0
        W0_vecinos = self.vecinos
        if word in self.vocab:
            pre_score = 0
            if word in W0.keys():
                check = label*W0[word]
                if check >= 0:
                    pre_score = np.tanh(beta1*(label + W0[word]))
                if check < 0:
                    x = label - W0[word]
                    if x <= 0:
                        pre_score = exp(beta2*x)
                    else: 
                        pre_score = -exp(-beta2*x)
            elif word in W0_vecinos.keys():
                check = label*W0[W0_vecinos[word][1]]
                if check >= 0:
                    pre_score = np.tanh(beta1*(label + (W0_vecinos[word][0]*W0[W0_vecinos[word][1]])))
                if check < 0:
                    x = label - (W0_vecinos[word][0]*W0[W0_vecinos[word][1]])
                    if x <= 0:
                        pre_score = exp(beta2*x)
                    else: 
                        pre_score = -exp(-beta2*x)
            return pre_score 

    def __count_freq(self):
        freqs = {w:0 for w in self.vocab}
        for j in self.data_df.index.to_list():
            BOW = self.data_df.iloc[j,0]
            for word in BOW.split():
                try:
                    freqs[word] += 1
                except:
                    pass
        return freqs

    def transform(self,df,beta1=1,beta2=1,text_col="text",label_col="label"):
        if self.W0_check:
            self.beta1 = beta1
            self.beta2 = beta2
            self.data_df = df[[text_col,label_col]].copy()
            self.data_df.rename(columns={text_col:"text",label_col:"label"},inplace=True)  # Estandarizar el nombre de las columnas del dataframe interno
            self.frequencies = self.__count_freq()
            y = self.data_df["label"].values
            self.n_classes = np.unique(y).shape[0]
            scoring = {w:0 for w in self.vocab} 
            #--- Acumulamos los scores:
            for j,k in enumerate(self.data_df.index.to_list()): 
                BOW = self.data_df.loc[k,"text"]
                BOW = BOW.split()
                for w in BOW:
                    scoring[w] += self.__s_tilde(y[j],w)
            #--- Normalizamos dividiendo entre la frecuencia:
            for w in scoring.keys():
                if self.frequencies[w]!=0:
                    scoring[w] = scoring[w]/self.frequencies[w]
                else:
                    scoring[w] = 0
            self.scoring = scoring
            #--- Construir el dataframe
            data = {'word':[w for w in self.vocab],
                    'score':[scoring[w] for w in self.vocab]
                    }
            scores_df = pd.DataFrame(data)
            self.scoring_check = True
            return scores_df
        else:
            self.build_neighbors()
            self.W0_check = True
            print("No se habían construido los vecinos más cercanos. Volver a ejecutar 'transform'.")
    
    def get_t_score(self):
        pass

    def change_model(self,new_model,kv='False'):
        self.model = new_model
        if kv:
            self.kv = True

    def get_words_representations(self,mode='mean'):
        '''
        Esta función regresa las representaciones de las palabras como re-escalamientos por el score de cada palabra. Hay varios modos:
        'mean': La representación de cada palabra es el promedio pesado de los pre-puntajes en cada etiqueta.
        '''
        if self.scoring_check and self.W0_check:
            N = len(self.vocab)
            n_dim = self.model.vector_size
            if self.kv:
                kv = self.model
            else:
                kv = self.model.wv
            X_word_rep = np.zeros(shape=(N,n_dim))
            for k,word in enumerate(self.vocab):
                try:
                    X_word_rep[k,:] = self.scoring[word]*kv.get_vector(word,norm=True) 
                    # X_word_rep[k,:] = self.scoring[word]*self.model.wv.get_vector(word,norm=True)
                except:
                    pass 
            self.word_reps = X_word_rep
            self.words_rep_check = True
            return X_word_rep
        else:
            print("No se ha ejecutado 'build_neighbors' o 'transform'.")
        
    def get_texts_representations_mean(self):
        '''
        Este método obtiene las representaciones de los textos, cada una es un vector, el cual es el promedio de las representaciones
        de las palabras que lo componen
        '''
        n_dim = self.model.vector_size
        if self.words_rep_check:
            X_word_rep = self.word_reps
        else:
            X_word_rep = self.get_words_representations()
        M = self.data_df.shape[0]
        X_msj_rep = np.zeros(shape=(M,n_dim))
        for k in self.data_df.index.to_list():
            rep_msj = np.zeros(shape=(n_dim,))
            BOW = self.data_df.loc[k,'text']
            for word in BOW.split():
                idx = self.vocab.index(word)
                rep_msj += X_word_rep[idx]
            rep_msj = rep_msj/len(BOW)
            X_msj_rep[k] = rep_msj
        self.text_rep = X_msj_rep
        return X_msj_rep

    def get_texts_representations_Nmean(self,n=3):
        '''
        'partial-mean': Lo mismo que <get_texts_representations_mean>, pero sólo se promedian las <n> palabras con score más alto, en valor absoluto.
        '''
        n_dim = self.model.vector_size
        if self.words_rep_check:
            X_word_rep = self.word_reps
        else:
            X_word_rep = self.get_words_representations()
        M = self.data_df.shape[0]
        X_msj_rep = np.zeros(shape=(M,n_dim))
        for k in self.data_df.index.to_list():
            mat_rep = np.zeros(shape=(n,n_dim))  # Matriz con las <n> representaciones vectoriales de las palabras, las más grandes en val. abs.
            BOW = self.data_df.loc[k,'text'].split()
            words_scores = [(word,self.scoring[word]) for word in BOW if word in self.vocab] # Juntamos los scores de cada palabra del msj
            non_zero_scores = [x for x in words_scores if x[1]!=0] # Sólo nos quedamos con las palabras del mensaje con pre-score diferente de 0
            size = len(non_zero_scores) 
            lista = [(x[0],abs(x[1])) for x in non_zero_scores] # Construimos una nueva lista con el valor abs de cada score y su idx en la lista original
            sorted_non_zero_scores = sorted(lista,key=itemgetter(1), reverse=True) # Ordenamos los valores absolutos de los scores de las palabras que forman el msj
            for i in range(min(n,size)):
                word_idx = self.vocab.index(sorted_non_zero_scores[i][0])
                mat_rep[i,:] = X_word_rep[word_idx]
            X_msj_rep[k,:] = np.mean(mat_rep,axis=0)
        self.text_rep = X_msj_rep
        return X_msj_rep

    def get_texts_representations_Wmean(self,weights):
        pass


    def get_texts_representations_MAT(self,cols_num=1):
        '''
        Esta función obtiene las representaciones de los mensajes, cada una de las cuales es una matriz conteniendo los k vectores (palabras)
        con el mayor pre-score, ya sea en valor absoluto o que conincidan con la polaridad del mensaje.
        <cols_num>: La cantidad de columnas que contiene la representación del mensaje
        El tensor que regresa es de tamaño (msg_num,100,cols_num)
        '''
        M = self.data_df.shape[0]
        n_dim = self.model.vector_size
        if self.kv:
            kv = self.model
        else:
            kv = self.model.wv
        X_msj_rep = np.zeros(shape=(cols_num,M,n_dim))
        for k in self.data_df.index.to_list():
            BOW = self.data_df.loc[k,"text"].split()
            words_scores = [(word,self.scoring[word]) for word in BOW] # Juntamos los scores de cada palabra del msj
            non_zero_scores = [x for x in words_scores if x[1]!=0] # Sólo nos quedamos con las palabras del mensaje con pre-score diferente de 0
            size = len(non_zero_scores) 
            lista = [(x[0],abs(x[1]),j) for j,x in enumerate(non_zero_scores)] # Construimos una nueva lista con el valor abs de cada pre-score y su idx en la lista original
            sorted_non_zero_scores = sorted(lista,key=itemgetter(1), reverse=True) # Ordenamos los valores absolutos de los pre-scores de las palabras que forman el msj
            A_m = np.zeros(shape=(n_dim,size)) # La matriz que contendrá como columnas las representaciones de las palabras
            if size>1:  # Si tiene más de una palabra con prescore distinto de 0, escogemos las mayores
                for j,triplet in enumerate(sorted_non_zero_scores):
                    try:
                        A_m[:,j] = non_zero_scores[triplet[2]][1]*kv.get_vector(non_zero_scores[triplet[2]][0],norm=True)
                    except:
                        pass
                if size>=cols_num:
                    A_m = A_m[:,:cols_num].copy() # Sólo nos quedamos con las primeras <cols_num> columnas
                    X_msj_rep[:,k,:] = A_m.reshape((A_m.shape[1],A_m.shape[0]))
                else:
                    X_msj_rep[:size,k,:] = A_m.reshape((A_m.shape[1],A_m.shape[0]))
            elif size==1: # Si sólo hay una palabra con prescore distinto de 0, esa es la representación del mensaje
                pair = non_zero_scores[0]
                try:
                    X_msj_rep[0,k,:] = pair[1]*kv.get_vector(pair[0],norm=True)  
                except:
                    pass
        if cols_num==1:
            X_msj_rep = X_msj_rep.reshape(M,n_dim)
        X_msj_rep = np.swapaxes(np.swapaxes(X_msj_rep,0,2),0,1)
        self.text_rep = X_msj_rep
        return X_msj_rep

    def get_texts_representations_PCA(self):
        '''
        Esta función obtiene las representaciones de los mensajes usando PCA en la matriz de representaciones de las palabras
        presentes en cada mensaje, las representaciones de las palabras es obtenida usando el score de cada palabra 
        '''
        N = len(self.vocab)
        n_dim = self.model.vector_size
        if self.words_rep_check:
            X_word_rep = self.word_reps
        else:
            X_word_rep = self.get_words_representations()
        M = self.data_df.shape[0]
        X_msj_rep = np.zeros(shape=(M,n_dim))
        for k in self.data_df.index.to_list():
            BOW = self.data_df.loc[k,'text'].split()
            A_m = np.zeros(shape=(n_dim,len(BOW)))
            for j,word in enumerate(BOW):
                idx = self.vocab.index(word)
                A_m[:,j] = X_word_rep[idx,:]
            pca = PCA(svd_solver='auto')
            A_m_pca = pca.fit_transform(A_m)
            X_msj_rep[k,:] = A_m_pca[:,0].copy()
        self.text_rep = X_msj_rep
        return X_msj_rep
    
    def plot_pca_texts_rep(self,img_label:str,save=False):
        '''
        Esta función grafica las representaciones vectoriales de los mensajes usando PCA como reducción de dimensionalidad.
        Cada etiqueta se representa con un color diferente. AVERIGUAR COMO REGRESAR LOS AXIS
        '''
        pca = PCA(svd_solver='auto')
        A_msj_pca = pca.fit_transform(self.text_rep)
        A_msj_pca_df = pd.DataFrame(A_msj_pca)
        labels = np.unique(self.data_df["label"].values)
        label_idxs = [self.data_df[self.data_df['label']==j].index.to_list() for j in labels]
        plt.figure(figsize=(7,5),dpi=500)
        title = "Representaciones de mensajes\n"+img_label
        plt.suptitle(title)
        for tag in labels:
            plt.scatter(x = A_msj_pca_df.loc[label_idxs[int(tag)],0], y = A_msj_pca_df.loc[label_idxs[int(tag)],1],alpha=0.5,
                        label=tag) 
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc='best')
        plt.axis("off")
        if save:
            fname = 'IMAGES/'+img_label+'texts-representations-PCA.png'
            plt.savefig(fname,dpi=500)
        plt.show()
    
    def plot_pca_words_rep(self,img_label:str,save=False):
        '''
        Esta función grafica las representaciones vectoriales de las palabras usando PCA como reducción de dimensionalidad.
        El color representa el score de la palabra
        '''
        if self.words_rep_check:
            scores = list(self.scoring.values())
            pca = PCA(svd_solver='auto')
            X_pca = pca.fit_transform(self.word_reps)
            X_pca_df = pd.DataFrame(X_pca)
            plt.figure(figsize=(7,5),dpi=500)
            title = "Representaciones de palabras\n"+img_label
            plt.suptitle(title)
            plt.scatter(x = X_pca_df[0].values, y = X_pca_df[1].values,alpha=0.75,
                        c = scores, cmap= 'RdBu') 
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(orientation="horizontal")
            plt.axis("off")
            if save:
                fname = 'IMAGES/'+img_label+'word-representations-PCA.png'
                plt.savefig(fname,dpi=500)
            plt.show()
        else:
            print("Correr primero <get_words_representations()>")

    def save_neighbors(self,fname):
        if self.W0_check:
            with open(fname, 'wb') as handle:
                pickle.dump(self.vecinos, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Neighbors dictionary saved.")
        else:
            print("Neighbors have not been calculated. Try <build_neighbors> first")

    def load_neighbors(self,fname):
        with open(fname, 'rb') as handle:
            self.vecinos = pickle.load(handle)
        self.W0_check = True

    def save_scoring(self,fname):
        if self.scoring_check:
            data = {'word':list(self.scoring.keys()),
                    'score':list(self.scoring.values())
                    }
            scoring_df = pd.DataFrame(data=data,columns=['word','score'])
            filename = fname + ".csv"
            scoring_df.to_csv(filename)
        else:
            print("Scoring has not been calculated. Try <transform> first.")

    def load_scoring(self,fname):
        scoring_df = pd.read_csv(fname,index_col=0)
        self.scoring = {scoring_df.loc[j,'word']:scoring_df.loc[j,'score'] for j in scoring_df.index.to_list()}
        self.scoring_check = True
        self.W0_check = True
            
