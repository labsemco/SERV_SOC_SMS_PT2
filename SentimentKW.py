import pandas as pd
import numpy as np
from TextRank import TextRank4Keyword
from sklearn.preprocessing import MinMaxScaler

class KW:

    def __init__(self,df,text_col_name:str,label_col_name:str):
        self.df = df.copy()
        self.text_col_name = text_col_name
        self.label_col_name = label_col_name
        # Obtener vocabulario:
        self.vocabulario = list(set(" ".join(list(self.df[text_col_name].values)).split()))
        # Obtener un diccionario de frecuencias de cada palabra en el vocabulario:
        self.__get_freqs()
        # Obtener las etiquetas diferentes, ordenadas de menor a mayor:
        x = self.df[self.label_col_name].unique()
        self.labels = sorted(x[~np.isnan(x)])
        self.normalized_labels = {x:(2/(len(self.labels)-1))*(x-min(self.labels)-len(self.labels)+1) + 1 for x in self.labels}

    def __get_freqs(self):
        self.frecuencias = {w:0 for w in self.vocabulario}
        word_sequence = " ".join(list(self.df[self.text_col_name].values)).split()
        for w in word_sequence:
            self.frecuencias[w] += 1

    def __get_t_score(self,word:str):
        '''
        Esta función calcula el t-score de una palabra <word>. 
        <freq>:         frecuencia de la palabra en el corpus
        <mensajes_df>:  dataframe con los mensajes
        <text_name>:    nombre de la columna donde está el texto procesado de los mensajes.
        <label_name>:   nombre de la columna donde está la etiqueta del mensaje.
        <labels>:       lista con los posibles valores de las etiquetas, con la forma [p_0,p_0+1,...,p_0+k]
        '''
        t_s = 0
        for label in self.labels:
            msjs = list(self.df[self.df[self.label_col_name]==label][self.text_col_name].values)
            textos = ' '.join(msjs)
            N = len([w for w in textos.split() if w==word]) # frecuencia de la palabra en mensajes con esta etiqueta
            normalized_label = (2/(len(self.labels)-1))*(label-min(self.labels)-len(self.labels)+1) + 1
            t_s += (normalized_label*N)
        t_s = t_s/self.frecuencias[word]
        return t_s

    def get_kw(self,method='TR',topn=100):
        # 1. Obtenemos las keywords de frecuencia en cada label
        t_scores = {w:self.__get_t_score(w) for w in self.vocabulario}   
        kw_scores_df = pd.DataFrame(data={
                'word':[w for w in self.vocabulario],
                't-damp-score':[t_scores[w]*(1-np.exp(1-self.frecuencias[w])) for w in self.vocabulario]
                })
        # 2A. Obtenemos las keywords de TextRank globales
        texto = " ".join([self.df.loc[j,'TEXTO_SIN_STOPWORDS'] for j in self.df.index.to_list()])
        global_tr_df = self.__get_TR_kw(texto,topn=topn)
        # 2B. Obtenemos las keywords de TextRank por etiqueta
        labels_tr_df = pd.DataFrame(data={'word':[],'TR index':[],'label':[]})
        for j in self.labels:
            idxs = self.df[self.df[self.label_col_name]==j].index.to_list()
            texto = " ".join([self.df.loc[j,self.text_col_name] for j in idxs])
            temp_tr_df = self.__get_TR_kw(texto,topn=topn)
            temp_tr_df['label'] = [int(j) for k in temp_tr_df.index.to_list()]
            labels_tr_df = labels_tr_df.append(temp_tr_df,ignore_index=True)
        # 2C. Obtenemos las keywords de TextRank propias de cada etiqueta
        labels_tr_df = self.__get_words_per_label(df_gral=global_tr_df,df_label=labels_tr_df)
        # 2D. Obtenemos un prescore para las keywords de TR, de acuerdo al label que representan
        scaler = MinMaxScaler(feature_range=(0,1))
        x = labels_tr_df['TR index'].values.reshape(-1, 1)
        scaler.fit(np.append(x,0).reshape(-1,1))
        labels_tr_df['scaled TR index'] = scaler.transform(labels_tr_df['TR index'].values.reshape(-1, 1))
        labels_tr_df['TR-score'] = labels_tr_df['scaled TR index']*[self.normalized_labels[x] for x in labels_tr_df['label'].values]
        # 3. Combinar ambos pre-scores
        prescores_df = kw_scores_df[['word','t-damp-score']].merge(
                                labels_tr_df[['word','TR-score']],
                                on='word',how='outer')
        prescores_df.fillna(value=0,inplace=True)
        prescores_df['combined-prescore'] = 0.5*(prescores_df['t-damp-score']+prescores_df['TR-score'])
        prescores_df = prescores_df[prescores_df['combined-prescore']!=0].copy()
        self.prescores_dict = {prescores_df.loc[j,'word']:prescores_df.loc[j,'combined-prescore'] 
                                    for j in prescores_df.index.to_list()}
        return self.prescores_dict

    def __get_TR_kw(self,texto:str,topn=100):
        TR = TextRank4Keyword()
        TR.analyze(texto, candidate_pos = ['NOUN','PROPN'], window_size=4, lower=False)
        TR_kw = TR.get_keywords(topn)
        return pd.DataFrame([[key, TR_kw[key]] for key in TR_kw.keys()], columns=['word', 'TR index'])

    def __get_words_per_label(self,df_gral,df_label):
        labels = []
        words = []
        values = []
        for j in self.labels:
            idxs = df_label[df_label['label']==j].index.to_list()
            all_words_in_label = [(j,w) for j,w in zip(idxs,list(df_label.loc[idxs]['word'].values))]
            gral_idxs = df_gral.index.to_list()
            label_words_idxs = [x[0] for x in all_words_in_label 
                                if x[1] not in list(df_gral.loc[gral_idxs]['word'].values) ]
            for k in label_words_idxs:
                labels.append(j)
                words.append(df_label.loc[k,'word'])
                values.append(df_label.loc[k,'TR index'])
        return pd.DataFrame(data={'word':words,'TR index':values,'label':labels})
    

