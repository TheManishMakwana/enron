
import re

class CleanUp():
    '''
    Basic cleanup utility class

    '''

    def __inti__(self):
        pass

    @classmethod
    def cleanEmail(self,email):

        email = re.sub(r"frozenset\({'",'',email)
        email = re.sub(r"\'}\)","",email)
        return email
    @classmethod
    def basicBodyClean(self,body_content):
        body_content = re.sub(r"\n+","\n",body_content)
        body_content = body_content.strip()
        return body_content
    @classmethod
    def cleanBody(self,body_content):
        body_content = body_content.lower()
        body_content = re.sub(r"\n"," ",body_content)
        body_content = re.sub(r"/HOU/ECT@ECT"," ",body_content)
        body_content = re.sub(r'\S+@\S+',' ',body_content) # removing email addresses
        body_content = re.sub(r'\w{3}-\w{3}-\w{4}',' ',body_content) # removing phone numbers
        body_content = re.sub(r"[0-9]{1,4}[\_|\-|\/|\|][0-9]{1,2}[\_|\-|\/|\|][0-9]{1,4}",' ',body_content) ## removing all dates
        body_content = re.sub(r'\W',' ',body_content)
        body_content = re.sub(r'\s\d+\s',' ',body_content) ## removing numbers
        body_content = re.sub(r'\s\d+',' ',body_content) ## removing numbers
        body_content = re.sub(r'^\d+\s',' ',body_content) ## removing numbers
        body_content = re.sub(r'\s+',' ',body_content) ## removing numbers

        body_content=body_content.strip()



        return body_content