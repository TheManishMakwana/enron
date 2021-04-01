import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

import pandas as pd
import nltk
from nltk import ngrams
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from wordcloud import WordCloud

portStemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
class TextTransformation():
    '''
    This class is utility class to perform any text transformation at rapid fast paced. Further functionality can be addedd for future projects and analysis need.

    '''
    def __init__(self):
        self.nlp = English()


    def remove_stopwords_from_bodyText(self,bodyText):
        doc = self.nlp(bodyText)
        ## word tokens without stop words
        keywords = ['http','www','ect','forwarded','thanks','thanks','subject','com']
        tokens = [word.text for word in doc if word.text not in STOP_WORDS and len(word.text)>=3 and word.text not in keywords]

        tokens_text = " ".join(tokens)
        return tokens_text


    def porter_stemming(self,token_text):
        tokens = token_text.split(' ')
        stemmed_tokens = [portStemmer.stem(word) for word in tokens]
        stemmed_tokens_text = " ".join(stemmed_tokens)
        return stemmed_tokens_text

    def lemmatizing(self,token_text):
        tokens = token_text.split(' ')
        lemmatized_tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
        lemmatized_tokens_text = " ".join(lemmatized_tokens)
        return lemmatized_tokens_text

    def createDTM_fromCorpus(self,dataframe,column_name,ngram_range=(1,2)):


        cv = CountVectorizer(stop_words=STOP_WORDS,ngram_range=ngram_range)
        data_cv = cv.fit_transform(dataframe[column_name])

        #creating DTM matrix
        data_dtm = pd.DataFrame(data_cv.toarray(),columns=cv.get_feature_names())
        data_dtm.index = dataframe.index
        return data_dtm,cv

    def createTFIDF_fromCorpus(self,dataframe,column_name,ngram_range=(1,2)):

        tfidfVect =TfidfVectorizer(stop_words=STOP_WORDS,ngram_range=ngram_range)
        data_cv = tfidfVect.fit_transform(dataframe[column_name])

        # creating matrix
        data_dtm = pd.DataFrame(data_cv.toarray(),columns=tfidfVect.get_feature_names())
        data_dtm.index = dataframe.index
        return data_dtm,tfidfVect

    def getTextBlob(self,text):

        wc = WordCloud(stopwords=STOP_WORDS,background_color='white',colormap='Dark2', max_font_size=80,random_state=12)
        wc.generate(text)
        return wc
