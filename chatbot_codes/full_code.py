import keras
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import random
import datetime
import webbrowser
import requests
import billboard
import time
from pygame import mixer
import COVID19Py
import joblib
import spacy


words=[]
classes=[]
documents=[]
ignore=['?','!',',',"'s"]

data_file=open('intents.json').read()
intents=json.loads(data_file)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = pattern.lower()
        doc = nlp(pattern)
        w = [token.lemma_ for token in doc] 
        words.extend(w)
        documents.append((w,intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words=[word for word in words if word not in ignore]

words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#training data
training=[]
output_empty=[0]*len(classes)

for doc in documents:
    bag=[]
    pattern=doc[0]

    p = []
    for pattern_ in pattern:
        doc_ = nlp(pattern_.lower())
        p.extend([token.lemma_ for token in doc_])

    pattern = p

    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    
    training.append([bag,output_row])
    
random.shuffle(training)
training=np.array(training)  
X_train=list(training[:,0])
y_train=list(training[:,1])  

#Model
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))

adam=keras.optimizers.Adam(0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)
weights=model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)    
model.save('mymodel.h5',weights)

from keras.models import load_model
model = load_model('mymodel.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


#Predict
def clean_up(sentence):
    # sentence_words=nltk.word_tokenize(sentence)
    sentence = sentence.lower()
    sentence_words = nlp(sentence)
    sentence_words = [word.lemma_ for word in sentence_words]
    return sentence_words

def create_bow(sentence,words):
    sentence_words=clean_up(sentence)
    bag=list(np.zeros(len(words)))
    
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence,model):
    p=create_bow(sentence,words)
    res=model.predict(np.array([p]))[0]
    threshold=0.8
    results=[[i,r] for i,r in enumerate(res) if r>threshold]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    
    for result in results:
        return_list.append({'intent':classes[result[0]],'prob':str(result[1])})
    return return_list

def get_response(return_list,intents_json):
    
    if len(return_list)==0:
        tag='noanswer'
    else:    
        tag=return_list[0]['intent']
    if tag=='datetime':        
        print(time.strftime("%A"))
        print (time.strftime("%d %B %Y"))
        print (time.strftime("%H:%M:%S"))


    list_of_intents= intents_json['intents']    
    for i in list_of_intents:
        if tag==i['tag'] :
            result= random.choice(i['responses'])
    return result

def response(text):
    return_list=predict_class(text,model)
    response=get_response(return_list,intents)
    return response

while(1):
    x=input()
    print(response(x))
    if x.lower() in ['bye','goodbye','get lost','see you']:  
        break


#Self learning
print('Help me Learn?')
tag=input('Please enter general category of your question  ')
flag=-1
for i in range(len(intents['intents'])):
    if tag.lower() in intents['intents'][i]['tag']:
        intents['intents'][i]['patterns'].append(input('Enter your message: '))
        intents['intents'][i]['responses'].append(input('Enter expected reply: '))        
        flag=1

if flag==-1:
    
    intents['intents'].append (
        {'tag':tag,
         'patterns': [input('Please enter your message')],
         'responses': [input('Enter expected reply')]})
    
with open('intents.json','w') as outfile:
    outfile.write(json.dumps(intents,indent=4))
