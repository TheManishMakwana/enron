from datetime import datetime
from ast import literal_eval
from cleaning import CleanUp
from collections import Counter
from textblob import TextBlob
from cleaning import CleanUp

from text_transformation import TextTransformation


import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

text_tranformer = TextTransformation()

class ShallowFeaures():
    '''
    This class is to create Shallow featured and add more columns with meaningful data for further analysis

    '''
    def __init__(self,df):
        self.data = df.copy()
        self.basic_cleanup()

        self.AddEmailDomain()
        self.AddDomainType()
        self.AddEmailType()
        self.AddEmailDateFeatures()
        self.AddBodyContentFeatures()

    def basic_cleanup(self):
        df = self.data.copy()
        df = df.fillna(' ')
        df['From'] =df['From'].apply(lambda x: CleanUp.cleanEmail(x))
        df['To'] =df['To'].apply(lambda x: CleanUp.cleanEmail(x))

        df['content'] = df['content'].apply(lambda x: CleanUp.basicBodyClean(str(x)))
        self.data= df

    @classmethod
    def from_dataframe(cls,dataframe):
        '''factory constrctor function to use as a interface to create Shallow Features and it's object'''
        df = dataframe.copy()

        sfObj = ShallowFeaures(df)
        return sfObj.data

    def AddEmailDomain(self):
        ''' Add Email domains as shallow feature'''
        df= self.data.copy()

        df = df.dropna(subset=['From','To'])

        df['IsFROMEmailDomainENRON'] = df['From'].apply(lambda x: 'YES' if "enron.com" in str(x).lower() else 'NO')
        df['IsTOEmailDomainENRON'] = df['To'].apply(lambda x: 'YES' if "enron.com" in str(x).lower() else 'NO')

        df['FROM_EMAIL_DOMAIN'] = df['From'].apply(lambda x: ".".join(str(x).split('@')[-1].split('.')[-2:]))
        df['TO_EMAIL_DOMAIN'] = df['To'].apply(lambda x: ".".join(str(x).split('@')[-1].split('.')[-2:]))

        self.data =df

    def AddDomainType(self):
        ''' Add Email domains type as shallow feature'''
        df= self.data.copy()

        df['FROM_EmailDomainType'] = df['From'].apply(lambda x: x.split('.')[-1])
        df['TO_EmailDomainType'] = df['To'].apply(lambda x: x.split('.')[-1])
        self.data =df

    def _getEmailType(self,subject):
        email_type = "NEW"
        if str(subject).lower().startswith("re:"): email_type="REPLY"
        if str(subject).lower().startswith("fw:"): email_type="FORWARD"

        return email_type
    def AddEmailType(self):
        ''' Add Email type as shallow feature'''
        df=self.data.copy()
        df['EMAIL_TYPE']=df['Subject'].apply(lambda x: self._getEmailType(x))
        df['EMAIL_TYPE'] = df[['EMAIL_TYPE','content']].apply(lambda x: 'FORWARD' if str(x[1]).startswith('---------------------- Forwarded by') else x[0],axis=1)
        self.data = df

    def AddEmailDateFeatures(self):
        df=self.data.copy()
        df['EMAIL_MONTH']=df['new_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%b'))
        df['EMAIL_YEAR']=df['new_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y'))
        df['EMAIL_MONTH_YEAR']=df['new_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%m-%Y'))
        df['EMAIL_DATE']=df['new_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
        self.data = df


    def _getHeader(self,x):
        content=x[0]
        subject=x[1]
        email_type=x[2]

        header=" "
        if email_type=='FORWARD':
            if subject.strip():
                header = str(content).split(str(subject))[0]
            else:
                header = str(content).split('Subject:')[0]
        return header

    def _getBody(self,x):
        content=x[0]
        subject=x[1]
        email_type=x[2]
        email_header=x[3]

        body=content
        if email_type=='FORWARD':
            if subject.strip():

                contentList = str(content).split(str(subject.strip()))
                body = " ".join(contentList[1:]) if len(contentList)>=2 else contentList[0]
            else:
                contentList = str(content).split('Subject:')
                body = " ".join(contentList[1:]) if len(contentList)>=2 else contentList[0]
            body = " " if body==email_header else body
        return body

    def AddBodyContentFeatures(self):
        df = self.data.copy()
        df['EMAIL_HEADER'] = df[['content','Subject','EMAIL_TYPE']].apply(lambda x: self._getHeader(x),axis=1)
        df['EMAIL_BODY'] = df[['content','Subject','EMAIL_TYPE','EMAIL_HEADER']].apply(lambda x: self._getBody(x),axis=1)

        df['CLEAN_BODY'] =df['EMAIL_BODY'].apply(lambda x: CleanUp.cleanBody(x))
        df['NO_OF_LINES'] = df['content'].apply(lambda x: len(str(x).split('\n')))
        df['EMAIL_LENGTH'] = df['content'].apply(lambda x: len(x))
        df['WORD_COUNT'] = df['content'].apply(lambda x: len(x.split(' ')))
        self.data =df

class DeepFeatures():
    '''
    This class is used to create Deep Features.

    '''
    def __init__(self,df):
        self.nlp = English()
        self.data = df
        self.AddUniqueWordCount()
        self.AddEmailSentiment()
        self.AddCleanTokens()
        self.AddDTM()

    @classmethod
    def from_dataframe(cls,dataframe):
        df = dataframe.copy()
        deepObj = DeepFeatures(df)
        return deepObj


    def AddUniqueWordCount(self):
        df = self.data.copy()
        df['UNIQUE_WORDS'] = df['CLEAN_BODY'].apply(lambda x: len(Counter(str(x).split()).most_common()))
        self.data = df

    def AddEmailSentiment(self):
        df = self.data.copy()
        df['SENTIMENT_POLARIT'] = df['EMAIL_BODY'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['SENTIMENT_SUBJECTIVITY'] = df['EMAIL_BODY'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        self.data = df

    def AddCleanTokens(self):
        df = self.data.copy()
        #df['TOKENS']= df['CLEAN_BODY'].apply(lambda x: self.remove_stopwords_from_bodyText(str(x)))
        df['TOKENS']= df['CLEAN_BODY'].apply(lambda x: text_tranformer.remove_stopwords_from_bodyText(str(x)))
        self.data=df
        self.email_corpus = " ".join(df['TOKENS'].tolist())

        df_email_type = df[['EMAIL_TYPE','TOKENS']].copy()
        self.new_email_corpus = " ".join(df[df['EMAIL_TYPE']=='NEW']['TOKENS'].tolist())
        self.reply_email_corpus = " ".join(df[df['EMAIL_TYPE']=='REPLY']['TOKENS'].tolist())
        self.forward_email_corpus = " ".join(df[df['EMAIL_TYPE']=='FORWARD']['TOKENS'].tolist())

    def AddDTM(self):
        df = self.data.copy()

        #df['STEM'] = df['TOKENS'].apply(lambda x: text_tranformer.porter_stemming(str(x)))

        self.DTM, self.CV = text_tranformer.createDTM_fromCorpus(df,'TOKENS')

