# Sentiment-Analysis-with-Word-Scoring

## 1. Información

Este repositorio describe el proceso completo para realizar análisis de sentimientos usando el puntaje y representación de palabras desarrollado y publicado en el artìculo [Cognitive Emotional Embedded Representations of Text to Predict Suicidal Ideation and Psychiatric Symptoms](https://www.mdpi.com/2227-7390/8/11/2088) . 

El módulo `SentimentKW` extrae palabras prototípicas de un dataframe de textos con sus respectivas etiquetas. La extracción se realiza en términos de frecuencia en cada etiqueta y TextRank. Al final se obtiene un diccionario de palabras prototípicas con pre-puntajes entre -1 y 1.

El módulo `scoring` contiene la clase que cálcula el puntaje de palabras y las representaciones de palabras y textos. Esto lo hace a partir de una lista de palabras prototípicas las cuales pueden ser definidas manualmente o automáticamente (usando el módulo `sentiment-kw`). En caso de ser necesario, la versión más reciente de este módulo, así como ejemplos de su uso, se encuentra en [este repositorio](https://github.com/labsemco/word-scoring).

El módulo `TextRank` contine la clase que extrae palabras clave usando TextRank.

## 2. Objetivo

El objetivo es explorar las diferentes columnas con etiquetas numéricas del dataframe y en cada una, explorar los diferentes métodos de representación de textos para obtener los resultados de clasificación (*accuracy* y *recall*) más altos. 
