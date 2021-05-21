import collections
import os
import math
import re
import codecs

def read_stopwords():
    if (os.path.isfile("stopwords.txt")) != True:
        print("Stopwords file not found!")
    else: 
        with open('stopwords.txt') as f:
            file = f.read().lower().replace('\n', ' ').split(' ')
        return file

def read_file(file,path):
    fhandler = codecs.open(path+"\\" + file,'rU','latin-1')
    find_words = re.findall('[A-Za-z0-9\']+', fhandler.read())
    words = list()
    for word in find_words:
        word = word.lower()
        words = words + [word]
    fhandler.close()    
    return words
    
def word_file_info(path):
    wordList = list()
    NumberOfFiles = 0
    for files in os.listdir(path):
        if files.endswith(".txt"):
            wordList = wordList + read_file(files,path)
            NumberOfFiles = NumberOfFiles + 1
    return wordList, NumberOfFiles

def ham_or_spam(ham_spam_bool):
    tot_files = spam_num + ham_num
    if ham_spam_bool == "ham":
        return ham_num/tot_files
    else:
        return spam_num/tot_files
    
def missing_word_count(word_dict,ham_spam_dict):
    for word in word_dict:
        if word not in ham_spam_dict:
            ham_spam_dict[word] = 0

def word_probability(classifier,remove_sw):
    count = 0
    if(remove_sw == 1):
            for word in stop_words:
                if word in ham_dict:
                    del ham_dict[word]
                if word in spam_dict:
                    del spam_dict[word]
                if word in bow_dict:
                    del bow_dict[word]                    
    if classifier == "ham":
        for word in ham_dict:
            count = count + (ham_dict[word] + 1)
        for word in ham_dict:
            ham_word_prob[word] = math.log((ham_dict[word] + 1)/count ,2)
    elif classifier == "spam":
        for word in spam_dict:
            count = count + (spam_dict[word] + 1)
        for word in spam_dict:
            spam_word_prob[word] = math.log((spam_dict[word] + 1)/count ,2) 

def ham_or_spam_selector(file_path, classifier):
    ham_prob = 0 
    spam_prob = 0 
    misclassified = 0
    num_files = 0
                   
    if classifier == "spam":
        for spam_file in os.listdir(file_path):
            words = read_file(spam_file,file_path)
            ham_prob = math.log(ham_or_spam("ham"),2)
            spam_prob = math.log(ham_or_spam("spam"),2) 
            for word in words:
                if word in ham_word_prob:
                    ham_prob = ham_prob + ham_word_prob[word]
                if word in spam_word_prob:
                    spam_prob = spam_prob + spam_word_prob[word]
            num_files =num_files + 1
            if(ham_prob >= spam_prob):
                misclassified = misclassified + 1
    if classifier == "ham":
        for ham_file in os.listdir(file_path):
            words = read_file(ham_file,file_path)
            ham_prob = math.log(ham_or_spam("ham"),2)
            spam_prob = math.log(ham_or_spam("spam"),2)           
            for word in words:
                if word in ham_word_prob:
                    ham_prob += ham_word_prob[word]
                if word in spam_word_prob:
                    spam_prob += spam_word_prob[word]
            num_files = num_files + 1
            if(ham_prob <= spam_prob):
                misclassified= misclassified + 1
    return misclassified,num_files

def calculations(ham_test, spam_test):
    misclassified_ham,tot_ham = ham_or_spam_selector(ham_test, "ham")
    misclassified_spam,tot_spam = ham_or_spam_selector(spam_test,"spam")
    ham_accuracy = ((tot_ham - misclassified_ham )/(tot_ham ))*100
    spam_accuracy = ((tot_spam -  misclassified_spam )/(tot_spam))*100
    emails_classified = tot_ham + tot_spam
    tot_misclassified = misclassified_ham + misclassified_spam
    accuracy = ((emails_classified  - tot_misclassified )/emails_classified)*100
    return ham_accuracy, spam_accuracy, accuracy

def print_results(h, s, t):
    print("Ham Accuracy: " + str(h))
    print("Spam Accuracy: " + str(s)) 
    print("Total Accuracy: " + str(t))


stop_words = read_stopwords()
train_path=input("Enter the path for the training data: ")
test_path=input("Enter the path for the testing data: ")
 
#train_path = assignment3_train/train
#test_path = assignment3_test/test"

ham_path = train_path + '/ham'
spam_path = train_path + '/spam'

ham_num, spam_num = 0, 0
ham_words = []
spam_words = []
ham_words,ham_num = word_file_info(ham_path)
spam_words,spam_num = word_file_info(spam_path)

ham_dict = dict(collections.Counter(word.lower() for word in ham_words))
spam_dict = dict(collections.Counter(word.lower() for word in spam_words))

bag_of_words = ham_words + spam_words
bow_dict = collections.Counter(bag_of_words)

missing_word_count(bow_dict,ham_dict)
missing_word_count(bow_dict,spam_dict)

ham_word_prob = dict()
spam_word_prob = dict()
           
word_probability("ham",0)
word_probability("spam",0) 

ham_test = test_path + '\ham'
spam_test = test_path + '\spam'


ham_accuracy, spam_accuracy, accuracy = calculations(ham_test, spam_test)

print("Implementation of Naive Bayes for text classification-")
print_results(ham_accuracy, spam_accuracy, accuracy)

word_probability("ham",1)
word_probability("spam",1) 
ham_accuracy, spam_accuracy, accuracy = calculations(ham_test, spam_test)

print("Implementation of Naive Bayes for text classification without stopwords-")
print_results(ham_accuracy, spam_accuracy, accuracy)

