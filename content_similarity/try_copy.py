import sys
import gensim
import sklearn
import numpy as np

from gensim.models.doc2vec import Doc2Vec, LabeledSentence
TaggededDocument=gensim.models.doc2vec.TaggedDocument

def get_data():
    with open("wiki.txt",'r', encoding='utf-8') as cf:
        docs =cf.readlines()
        print(len(docs))

    x_train=[]
    for i, text in enumerate(docs):
        word_list=text.split(" ")
        l = len(word_list)
        document =TaggededDocument(word_list, tags=[i])
        print("get_data:",i)
        x_train.append(document)


    return x_train


def getVecs(model, corpus, size):
    vecs=[np.array(model.docvecs[z.tag[0]].reshape(1,size)) for z in corpus]
    print("corpus:",z)
    return np.concatenate(vecs)

def train(x_train,size=200,epoch_num=1):
    print('start training')
    model_dm=Doc2Vec(x_train,min_count=1,window=3,size=size,sample=1e-3,negative=5,workers=4)
    print('rest')
    model_dm.train(x_train,total_examples=model_dm.corpus_count,epochs=70)
    
    print('trained')
    model_dm.save('model_yesyesyes')

    return model_dm

def test():
    model_dm=Doc2Vec.load('model_yesyesyes')
    test_text=['晚清','覺得','晚清','很','民族']
    inferred_vector_dm=model_dm.infer_vector(test_text)
    print(inferred_vector_dm)
    sims=model_dm.docvecs.most_similar([inferred_vector_dm],topn=3)

    return sims



x_train=get_data()
model_dm=train(x_train)

sims=test()
for count,sim in sims:
    sentence=x_train[count]
    words=''
    for word in sentence[0]:
        words=words+word+""
    print(words,sim,len(sentence[0]))
