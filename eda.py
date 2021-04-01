from collections import Counter
from text_transformation import TextTransformation

from gensim import matutils,models
import scipy.sparse

class EDA():
    '''
    This class is interface between data and business analysis. This can be enahced to add more business insights for end user to use.

    '''
    def __init__(self):
        pass

    @classmethod
    def top_sender(self,df,top_count=5):
        ''' Interface to get Top Sender information'''
        top_df = df.groupby('From')['content'].count().reset_index().sort_values(by=["content"], ascending=False)

        top_df=top_df.reset_index(drop=True)
        top_df.columns=['SENDER','EMAIL_COUNT']

        top_df=top_df.head(top_count).copy()

        top_df['SENDER_NAME']=top_df['SENDER'].apply(lambda x: " ".join(str(x).split('@')[0].split('.')))

        return top_df

    @classmethod
    def top_receiver(self,df,top_count=5):
        ''' Interface to get Top Receiver information'''
        top_df = df.groupby('To')['content'].count().reset_index().sort_values(by=["content"], ascending=False)

        top_df=top_df.reset_index(drop=True)
        top_df.columns=['RECEIVER','EMAIL_COUNT']

        top_df=top_df.head(top_count).copy()

        top_df['RECEIVER_NAME']=top_df['RECEIVER'].apply(lambda x: " ".join(str(x).split('@')[0].split('.')))
        return top_df

    @classmethod
    def top_sender_receiver_pair(self,df,top_count=5):
        ''' Interface to get Top Sender-Receiver pair information'''
        top_df = df.groupby(['From','To'])['content'].count().reset_index().sort_values(by=["content"], ascending=False)

        top_df=top_df.reset_index(drop=True)
        top_df.columns=['SENDER','RECEIVER','EMAIL_COUNT']

        top_df=top_df.head(top_count).copy()

        top_df['SENDER_NAME']=top_df['SENDER'].apply(lambda x: " ".join(str(x).split('@')[0].split('.')))
        top_df['RECEIVER_NAME']=top_df['RECEIVER'].apply(lambda x: " ".join(str(x).split('@')[0].split('.')))
        top_df['RECEIVER_DOMAIN']=top_df['RECEIVER'].apply(lambda x: ".".join(str(x).split('@')[-1].split('.')[-2:]))

        return top_df

    @classmethod
    def top_sender_receiver_domain_pair(self,df,top_count=5):
        ''' Interface to get Top Sender Receiver Domain pair information'''
        top_df = df.groupby(['FROM_EMAIL_DOMAIN','TO_EMAIL_DOMAIN'])['content'].count().reset_index().sort_values(by=["content"], ascending=False)

        top_df=top_df.reset_index(drop=True)
        top_df.columns=['FROM_EMAIL_DOMAIN','TO_EMAIL_DOMAIN','EMAIL_COUNT']

        top_df=top_df.head(top_count).copy()

        return top_df

    @classmethod
    def sender_sentiment_rate(self,df):
        ''' Interface to get Sentiment information and it's mean '''
        s_agg = {
            'SENTIMENT_POLARIT':['mean'],
            'SENTIMENT_SUBJECTIVITY':['mean']
        }

        top_df = df.groupby('From').agg(s_agg).reset_index()

        top_df=top_df.reset_index(drop=True)
        top_df.columns=['SENDER','SENTIMENT_POLARIT_MEAN','SENTIMENT_SUBJECTIVITY_MEAN']

        #top_df=top_df.head(top_count).copy()

        top_df['SENDER_NAME']=top_df['SENDER'].apply(lambda x: " ".join(str(x).split('@')[0].split('.')))
        top_df['SENDER_DOMAIN']=top_df['SENDER'].apply(lambda x: ".".join(str(x).split('@')[-1].split('.')[-2:]))

        return top_df

    @classmethod
    def top_words_used_in_communication(cls,email_corpus,top_count=50):
        ''' Interface to get Top words used in communication'''
        tokens = email_corpus.split(' ')
        return Counter(tokens).most_common(top_count)

    @classmethod
    def get_topwords_wordcloud(cls,email_corpus,top_count=50):
        ''' Interface to get word cloud for display'''
        tokens = email_corpus.split(' ')
        top_words = [w for (w,c) in Counter(tokens).most_common(top_count)]
        top_words_text = " ".join(top_words)
        text_transforer = TextTransformation()
        wordCloud = text_transforer.getTextBlob(top_words_text)
        return wordCloud

    @classmethod
    def topic_modeling_with_LDA(cls,DTM,CV,topic_range=range(2,5)):
        ''' Interface to get perform LDA - Topic modeling'''

        tdm = DTM.transpose()

        # keeping term doc matrix in Gensim format
        sparse_counts = scipy.sparse.csr_matrix(tdm)
        corpus = matutils.Sparse2Corpus(sparse_counts)

        id2word = dict((v,k) for k,v in CV.vocabulary_.items())

        for tp in topic_range:
            print(f'TOPIC Count- {tp} : ',end=' ')
            print('#*'*50)
            lda = models.LdaModel(corpus=corpus,id2word=id2word,num_topics=tp,passes=10)
            for topic in lda.print_topics():
                print('-'*50)
                print(topic)



