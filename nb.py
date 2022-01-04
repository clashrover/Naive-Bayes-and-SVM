from ut import *
import re
import math
import random
import numpy as np
import sys

def readData(fname):                
    # to read json file
    # convert dictionary to list of (star,review) pair
    l = json_reader(fname)
    k=0
    data = []
    for w in l:
        # print(w)
        stars=0
        text = ""
        for x in w:
            if x=="stars":
                stars = w[x]
            if x=="text":
                text += w[x]
                break
        data.append((stars,text))
    return data


def stemData(fname):
    # this function reads json file
    # removes stops words
    # does stemming
    # return list of (star,review)
    
    l = json_reader(fname)
    # k=0
    data = []
    star=0
    for w in l:
        text = None
        for x in w:
            if x=="stars":
                star = int(float(w[x]))
            if x=="text":
                text = getStemmedDocuments(w[x],return_tokens=False)
                # print(text)
                # print(k)
                # k+=1
                break
        data.append((star,text))
    
    return data


def getLabelParameters(data):
    m = len(data)
    y = [0.0]*6
    for i in range(m):
        p = data[i]
        x = int(float(p[0]))
        y[x]+=1.0
    
    y[:] = [x/m for x in y]
    return y 



def getWordParameter(data):
    word_info = {}
    num_words_class=[0]*6
    total_unique_words = 0
    for review in data:
        r = review[1]
        rl = re.findall(r'\w+', r) # use regex to find words in document
        star = int(float(review[0]))
        for word in rl:
            num_words_class[star]+=1
            if word in word_info:
                word_info[word][star]+=1
            else:
                total_unique_words+=1
                l= [0]*6
                l[star]=1
                word_info[word] = l

    return word_info, num_words_class, total_unique_words


def getBiGramParameter(data):
    biGram_info = {}
    num_biGrams_class=[0]*6
    total_unique_biGrams = 0
    for review in data:
        r = review[1]
        rl = re.findall(r'\w+', r) # use regex to find words in document
        star = int(float(review[0]))
        for i in range(len(rl)-1):
            biGram = rl[i]+ " " + rl[i+1]
            num_biGrams_class[star]+=1
            if biGram in biGram_info:
                biGram_info[biGram][star]+=1
            else:
                total_unique_biGrams+=1
                l= [0]*6
                l[star]=1
                biGram_info[biGram] = l

    return biGram_info, num_biGrams_class, total_unique_biGrams



def predict_word(s,word_info, num_words_class, total_unique_words, phi):
    rl = re.findall(r'\w+', s)      # extract words out of string
    l = [0]*6                       # probability of this string s belonging to class k
    l[1]+=math.log(phi[1])
    l[2]+=math.log(phi[2])
    l[3]+=math.log(phi[3])
    l[4]+=math.log(phi[4])
    l[5]+=math.log(phi[5])
    for word in rl:
        for i in range(6):
            if word in word_info:
                p = (word_info[word][i]+1) / (num_words_class[i]+ total_unique_words)
                p = math.log(p)
                l[i]+=p
    l.pop(0)
    return l.index(max(l))+1

def predict_biGram(s,biGram_info, num_biGrams_class, total_unique_biGrams, phi):
    rl = re.findall(r'\w+', s)      # extract biGrams out of string
    l = [0]*6                       # probability of this string s belonging to class k
    l[1]+=math.log(phi[1])
    l[2]+=math.log(phi[2])
    l[3]+=math.log(phi[3])
    l[4]+=math.log(phi[4])
    l[5]+=math.log(phi[5])
    for i in range(len(rl)-1):
        biGram = rl[i]+" " +rl[i+1]
        for i in range(6):
            if biGram in biGram_info:
                p = (biGram_info[biGram][i]+1) / (num_biGrams_class[i]+ total_unique_biGrams)
                p = math.log(p)
                l[i]+=p
    l.pop(0)
    return l.index(max(l))+1



def main(stem, feature_type):
    train_data = None
    if stem == 0:    # this stem is for no stemming or removing stop words
        train_data = readData(sys.argv[1])  # read the json file, convert it into dictionary of stars,text
    else:
        # print("Begin Stemming")
        train_data = stemData(sys.argv[1])
        # print("Done Stemming")
    
    phi = getLabelParameters(train_data) # to get a list of parameters related to label, that is stars.
    
    feature_info=None
    num_features_class=None 
    total_unique_features=None
    if feature_type == 0:                           #feature is single word
        feature_info, num_features_class, total_unique_features = getWordParameter(train_data)
    else:                                #feature is bi-gram (two consecuted words)
        feature_info, num_features_class, total_unique_features = getBiGramParameter(train_data)

    # feature_info is a dictionary that contains info about word and its occurance in each class
    # num_features_class contains info how many words in each class
    # total_unique words is size of word dictionary (bag of words), used in laplace smothing
    # to compute 0_{j=l/k} use (feature_info[l][k]+1)/(num_features_class[k]+ total_unique_features) with lapalace smothing there

    # prediction on train.json
    # score_train=0
    # mc = [0]*6
    # for review in train_data:
    #     s = review[1]
    #     label_predict = None
    #     if feature_type == 0:
    #         label_predict = predict_word(s,feature_info, num_features_class, total_unique_features, phi)
    #     else:
    #         label_predict = predict_biGram(s,feature_info, num_features_class, total_unique_features, phi)

    #     label = int(float(review[0]))
    #     mc[label]+=1
    #     if label_predict == label:
    #         score_train+=1
    
    # print("Accuracy on training using NB:",score_train/len(train_data))

    # prdiction on test.json
    test_data = None
    if stem == 0:
        test_data = readData(sys.argv[2])  # read the json file, convert it into dictionary of stars,text
    else:
        # print("Begin Stemming")
        test_data = stemData(sys.argv[2])
        # print("Done Stemming")
    
    score_test=0
    score_random =0
    score_majority = 0
    # major_label = mc.index(max(mc))
    
    confusion_matrix = np.zeros((5,5))

    output = open(sys.argv[3], 'w')
    
    for review in test_data:
        s = review[1]
        label_predict =None
        if feature_type == 0:
            label_predict = predict_word(s,feature_info, num_features_class, total_unique_features, phi)
        else:
            label_predict = predict_biGram(s,feature_info, num_features_class, total_unique_features, phi)
        
        print(int(label_predict),file=output)
        label = int(float(review[0]))

        confusion_matrix[label-1][label_predict-1]+=1.0

        label_random = random.randint(1,5)
        if label_predict == label:
            score_test+=1
        if label_random == label:
            score_random+=1
        # if major_label == label:
            # score_majority+=1
        

    # print("Accuracy on test using NB:",score_test/len(test_data))
    if stem == 0:
        # print("Accuracy on test using Random pick:",score_random/len(test_data))
        # print("Accuracy on test using Majority class:",score_majority/len(test_data))
        # print(confusion_matrix)
        m=0.0
        x=0
        for i in range(5):
            if m < confusion_matrix[i][i]:
                m = confusion_matrix[i][i]
                x = i
        # print("Higest diagonal value class :",x+1)



main(0, 0)

# Without stemming results

# single word
# Accuracy on training using NB: 0.6429725242674883
# Accuracy on test using NB: 0.6061412824002752


# biGram
# Accuracy on training using NB: 0.8431306929508369
# Accuracy on test using NB: 0.6407439536935939
# Accuracy on test using Random pick: 0.19907566670156598
# Accuracy on test using Majority class: 0.439895900327555
# [[1.6981e+04 5.8200e+02 5.5600e+02 1.3210e+03 7.2900e+02]
#  [4.2960e+03 6.2500e+02 1.4890e+03 3.6490e+03 7.7900e+02]
#  [1.6730e+03 1.4500e+02 1.3950e+03 9.5890e+03 1.7290e+03]
#  [7.0000e+02 1.6000e+01 2.1700e+02 1.7040e+04 1.1385e+04]
#  [9.9900e+02 7.0000e+00 6.1000e+01 8.1170e+03 4.9638e+04]]
# Higest diagonal value class : 5




# upon stemming results

# single word
# Begin Stemming
# Done Stemming
# Accuracy on training using NB: 0.6236351874841084
# Accuracy on test using NB: 0.596494114479726

# bi grams
# Begin Stemming
# Done Stemming
# Accuracy on training using NB: 0.8962349870623252
# Accuracy on test using NB: 0.6343424221122063
