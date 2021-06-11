import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from nltk.stem import PorterStemmer
import operator
from collections import Counter
import re
import math
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,manhattan_distances
import numpy as np
from nltk.stem import WordNetLemmatizer
 
wnl = WordNetLemmatizer()


df=pd.read_csv('chatbot/Faqs_pdeu.csv')
df.head()

def get_euclid(a,b):
    return math.sqrt(sum((a[k] - b[k])**2 for k in set(a.keys()).intersection(set(b.keys()))))
def get_man(a,b):
    return (sum((a[k] - b[k])**2 for k in set(a.keys()).intersection(set(b.keys()))))
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


WORD = re.compile(r"\w+")
f=open('chatbot/pdeu.txt','r',encoding='utf-8',errors='ignore')
raw=f.read()
raw=raw.lower()
#print(raw)
sent_tokens=nltk.sent_tokenize(raw)
# print(sent_tokens)
sent_tokens=[x.replace('\n','') for x in sent_tokens]
#print('------sent_tokens-----')
#print(sent_tokens)

word_tokens=nltk.word_tokenize(raw)
lemmer=nltk.stem.WordNetLemmatizer()
#print(sent_tokens)
#print(len(sent_tokens))

def lemmatize(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)
def normalize(text):
    return lemmatize(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def greet(sent):
    greet_resp=["hello welcome!!","hi how are you?","Pleasure to hear from you!!!","Hello sir","nice to  meet you sir!!!","What can I do for you?"]
    greet_inp=['hii','heyaa','hello','hey there',"hi","hey","hello","howdy","how are you?"]
    if sent in ["good morning","good afternoon","good evening"]:
        return f"hello , {sent}"
    if sent=="good night":
        return "good night"

    if(sent[-1]=='?'):
        sent=sent[:-1]
    ps = PorterStemmer()
    arr=sent.split(' ')
    arr=[ps.stem(i) for i in arr]
    print('\n\n----------------------------------',arr,'\n\n')
    if('see' and 'you') in arr:
        return 'Talk to you Later'
    elif 'goodby' in arr or 'bye' in arr:
        return 'Good Bye :)'
    elif 'accredit' in arr and 'colleg' in arr:
        return 'Yes'
    elif 'instal' in arr and 'fee'  in arr and 'pay' in arr:
        return 'Yes You can pay fees in two installmensts'
    elif 'hour' in arr and ('work'  in arr or 'oper'  in arr):
        return 'We are open 9:00am-4:00pm Monday-friday!'
    elif ('field' in arr or 'branch' in arr) and 'different' in arr and 'colleg' in arr:
        return '"Petroleum Technology-120,Mechanical Engineering-120,Electrical Engineering-120,Civil Engineering-120,Chemical Engineering-120,Computer Science-60,Information and Communication Technology-60".'
    elif ('cse' in arr or 'mechan' in arr or 'chemica' in arr or 'electr' in arr or 'comput' in arr or 'scienc' in arr or 'inform' or 'commun' in arr or 'technolg' in arr or 'petroleum' in arr) and 'subject' in arr:
        return 'You can check all this course related information from our website !'
    elif 'payment' in arr and 'fee'  in arr and 'avail' in arr:
        return 'cheque,debit card,netbanking,credit card are acceptable. NEFT is preferable'
    elif 'is' in arr and 'transportation' in arr and 'avail' in arr:
        return 'Yes , bus service is available.'
    elif 'hostel' in arr and 'facil' in arr and 'avail' in arr:
        return 'Yes! we provide telephone , internet , AC , first-aid , reading , dining , security all this facility in hostel'
    elif 'transportation' in arr and 'fee' in arr:
        return 'transportaion fees of our college is 10500 per semester'
    elif 'semest'in arr and 'fee' in arr:
        return 'fees of our college is 110000 per semester!'
    elif 'chairman' in arr and 'who' in arr and 'colleg' in arr:
        return  'Mukesh Ambani is chairman of our college'
    elif 'is' in arr and 'under' in arr and 'gtu' in arr:
        return 'No, our college doesnt come under GTU.'
    elif 'scholarship' in arr and 'criteria' in arr:
        return 'you can check out at :: https://www.pdpu.ac.in/downloads/Financial%20Assistance%202019.pdf'

    for word in sent.split():
        if word.lower() in greet_inp:
            return random.choice(greet_resp)
    return None

#Searching in file
# Response for searching in file using TF-IDF
def resp(user_inp):
    ans,ind,hue=[],[],3
    tfidvec=TfidfVectorizer(tokenizer=normalize,stop_words='english')
    tfid=tfidvec.fit_transform(sent_tokens)

    vals=cosine_similarity(tfid[-1],tfid)
    d={}
    for i in range(0,len(vals[0])):
    	d[i]=vals[0][i]
    sorted_d = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))
    for (key,val) in sorted_d.items():
    	if(hue>0 and val>0):
    		ind.append(key)
    	else:
    		break
    	hue-=1
    flat=vals.flatten()
    
    flat=sorted(flat,reverse=True)
    req_tfid=flat[0]
    if(req_tfid==0):
        ans=ans+"I am sorry! I don't understand you"    
    else:
        for index in ind: 
            ans.append(sent_tokens[index])
    ans1=''
    for statements in ans:
        ans1=ans1+str(statements)
        ans1+='\n'
    return ans1

def clean_sent(sent,stopwords=False):
  sent=sent.lower().strip()
  sent=re.sub(r'[^a-z0-9\s]','',sent)
  if stopwords:
    sent=remove_stopwords(sent)
  return sent 

def get_clean_sent(df,stopwords=False):
  sents=df[['Questions']]
  cleaned_sent=[]
  for index,row in df.iterrows():
    cleaned=clean_sent(row['Questions'],stopwords)
    cleaned=cleaned.lower()
    cleaned_sent.append(" ".join([wnl.lemmatize(i) for i in cleaned.split()]))
  return cleaned_sent

#Glove model
def getwordvec(word,model):
    samp=model['computer']
    sample_len=len(samp)
    vec=[0]*sample_len
    try:
        vec=model[word]
    except:
        vec=[0]*sample_len
    return vec

def getphrase(phrase,embeddingmodel):
    samp=getwordvec('computer',embeddingmodel)
    vec=np.array([0]*len(samp))
    den=0
    for word in phrase.split():
        den+=1
        vec=vec+np.array(getwordvec(word,embeddingmodel))
    return vec.reshape(1,-1)

def glove(question,cleaned_sent,param):
    google_model=gensim.models.KeyedVectors.load('chatbot/w2vecmodel.mod')
    sent_embedings=[]
    try_flag=False
    for sent in cleaned_sent:
        sent_embedings.append(getphrase(sent,google_model))
    ques_em=getphrase(question,google_model)
    max_sim=-1
    index_sim=-1
    try:
        for index,faq_em in enumerate(sent_embedings):
            if(param=='cosine'):
                sim=cosine_similarity(faq_em,ques_em)[0][0]
            if(param=='euclid'):
                sim=euclidean_distances(faq_em,ques_em)[0][0]
            if(param=='man'):
                sim=manhattan_distances(faq_em,ques_em)[0][0]         
            if(sim>max_sim):
                max_sim=sim
                index_sim=index
        try_flag=True
        ans=df.iloc[index_sim,1]
        return ans,try_flag
    except Exception as e:
        return 0,try_flag


#Response for bagofwords approach
def resp1(ques,param):
    cleaned_sent=get_clean_sent(df,stopwords=True)
    sentences=cleaned_sent
    sent_words=[[wrd for wrd in document.split()]for document in sentences]
    dictionary=corpora.Dictionary(sent_words)
    bow_corpus=[dictionary.doc2bow(text) for text in sent_words]
    ques=clean_sent(ques,stopwords=True)
    #print(ques)
    ques_em=dictionary.doc2bow(ques.split())
    #print(ques_em)
    ans,try_flag=glove(ques,cleaned_sent,param)
    #print('Returned ans :: ',ans)
    #print('try_flag :: ',try_flag)
    if try_flag:
        return ans
    return retrieve(ques_em,bow_corpus,df,sentences,ques,param)


def retrieve(ques_em,sent_em,df,sent,user_inp,param):
    max_sim=-1
    index_sim=-1
    try:
        for index,faq_em in enumerate(sent_em):
            if(param=='cosine'):
                sim=cosine_similarity(faq_em,ques_em)[0][0]
            if(param=='euclid'):
                sim=euclidean_distances(faq_em,ques_em)[0][0]
            if(param=='man'):
                sim=manhattan_distances(faq_em,ques_em)[0][0]         
            if(sim>max_sim):
                max_sim=sim
                index_sim=index
        ans3=df.iloc[index_sim,1]
        return ans3
    except Exception as e:
        pass
    ans1=resp(user_inp)
    ans2=search_google(user_inp)
    cos1,cos2=0,0
    inp=text_to_vector(user_inp)
    cos1=get_cosine(inp,text_to_vector(ans1))
    cos2=get_cosine(inp,text_to_vector(ans2))
    if(cos1>=cos2):
        return ans1
    return ans2

def get_bot_resp(user_inp,param):
    flag=False
    while(1):
        ans=greet(user_inp.lower())
        print("got ans for query",ans,user_inp)
        if(user_inp=='what are branches in sot'):
            ans="Following are the branches : Electrical,Chemical,Mechanical,Civil,Computer,ICT"
            flag=True
            return ans,flag
        if(user_inp=='is there hostel facility in pdeu'):
            ans="Yes there is hostel facility in pdeu"
            flag=True
            return ans,flag
        if(user_inp=='average fee per year'):
            ans='Average Fees 2,43,250 ruppes per year'
            flag=True
            return ans,flag
        if(ans!=None):
            flag=True
            return ans,flag
        return resp1(user_inp.lower(),param),flag



