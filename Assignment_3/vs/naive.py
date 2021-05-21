import collections
import os
import sys
import math

import add_files

#from add_files import word_file_info
#from add_files import ReadFile
#from add_files import stop_words



train_path=input("Enter the path for the training data: ")
test_path=input("Enter the path for the testing data: ")
 
#train_path = "assignment3_train/train"
#test_path = "assignment3_test/test"
#We need to find 
#P(ham) = number of documents belonging to category ham / Total Number of documents
#P(spam) = number of documents belonging to category spam / Total Number of documents
#P(ham|bodyText) = (P(ham) * P(bodyText|ham)) / P(bodyText)
#P(bodyText|spam) = P(word1|spam) * P(word2|spam)*.....
#P(bodyText|ham) = P(word1|ham) * P(word2|ham)*.....
#P(word1|spam) = count of word1 belonging to category spam / Total count of words belonging to category spam 
#P(word1|ham) = count of word1 belonging to category ham / Total count of words belonging to category ham    
#For new word not seen yet in test document
#P(new-word|ham) or P(new-word|spam) = 0
#This will make the product zero so we can solve this
# if (log(P(ham|bodyText)) > log(P(spam|bodyText)))
#    return 'ham'
#else:
#    return 'spam'
#    
#log(P(ham|bodyText)) = log(P(ham)) + log(P(bodyText|ham))
#                     = log(P(ham)) + log(P(word1|ham)) + log(P(word2|ham)) + .....
    
#P(word1|ham) = (count of word1 belonging to category ham + 1)/
#              (total number of words belonging to ham + number of distinct words in training database)
#P(word1|spam) = (count of word1 belonging to category spam + 1)/
#                 (total number of words belonging to spam + number of distinct words in training database)    
#location of the folder for ham & spam for train and test

ham_path = train_path + '/ham'
spam_path = train_path + '/spam'

#Find out all the words in ham folder and spam folder and find there counts  
ham_num, spam_num = 0, 0
ham_words = []
spam_words = []
ham_words,ham_num = word_file_info(ham_path)
spam_words,spam_num = word_file_info(spam_path)

#Function to Find out P(ham) and P(spam), by calculating
#the number of ham/spam documents and total number of documents
def ham_or_spam(ham_spam_bool):
    tot_files = spam_num + ham_num
    if ham_spam_bool == "ham":
        return ham_num/tot_files
    else:
        return spam_num/tot_files
        

#After finding all the words in ham and spam files , we will find the distinct words and its count
ham_dict = dict(collections.Counter(word.lower() for word in ham_words))
spam_dict = dict(collections.Counter(word.lower() for word in spam_words))

#making bag of words for both ham and spam and further counting the count of each Distinct word in it
bag_of_words = ham_words + spam_words
bow_dict = collections.Counter(bag_of_words)

def missing_word_count(word_dict,ham_spam_dict):
    for word in word_dict:
        if word not in ham_spam_dict:
            ham_spam_dict[word] = 0
            
#getting missing words in each Ham and Spam list and adding them and intializing their count= 0
missing_word_count(bow_dict,ham_dict)
missing_word_count(bow_dict,spam_dict)

#P(word1|ham) = (count of word1 belonging to category ham + 1)/
#              (total number of words belonging to ham + number of distinct words in training database)
#P(word1|spam) = (count of word1 belonging to category spam + 1)/
#                 (total number of words belonging to spam + number of distinct words in training database)  
#Here, Counter contains total number of words belonging to ham/spam plus number of distinct words 
#in training dataset as we updated all the missing words in dictionary too
ham_word_prob = dict()
spam_word_prob = dict()
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
            count += (ham_dict[word] + 1)
        for word in ham_dict:
            ham_word_prob[word] = math.log((ham_dict[word] + 1)/count ,2)
    elif classifier == "spam":
        for word in spam_dict:
            count += (spam_dict[word] + 1)
        for word in spam_dict:
            spam_word_prob[word] = math.log((spam_dict[word] + 1)/count ,2) 
           
#caluculating probability for each word in ham and Spam folders 
word_probability("ham",0)
word_probability("spam",0) 


#Finally classify the emails as ham or spam    
def ham_or_spam_selector(file_path, classifier):
    ham_prob = 0 
    spam_prob = 0 
    misclassified = 0
    num_files = 0
                   
    if classifier == "spam":
        for spam_file in os.listdir(file_path):
            words =ReadFile(spam_file,file_path)
            #find actual P(ham) and P(spam) i.e. (number of ham documents / Total no of documents)
            ham_prob = math.log(ham_or_spam("ham"),2)
            spam_prob = math.log(ham_or_spam("spam"),2)
            #log(P(ham|bodyText)) = log(P(ham)) + log(P(word1|ham)) + log(P(word2|ham)) + .... 
            for word in words:
                if word in ham_word_prob:
                    ham_prob += ham_word_prob[word]
                if word in spam_word_prob:
                    spam_prob += spam_word_prob[word]
            num_files +=1
            if(ham_prob >= spam_prob):
                misclassified+=1
    if classifier == "ham":
        for ham_file in os.listdir(file_path):
            words =ReadFile(ham_file,file_path)
            #find actual P(ham) and P(spam) i.e. (number of ham documents / Total no of documents)
            ham_prob = math.log(ham_or_spam("ham"),2)
            spam_prob = math.log(ham_or_spam("spam"),2)
            #log(P(ham|bodyText)) = log(P(ham)) + log(P(word1|ham)) + log(P(word2|ham)) + ....            
            for word in words:
                if word in ham_word_prob:
                    ham_prob += ham_word_prob[word]
                if word in spam_word_prob:
                    spam_prob += spam_word_prob[word]
            num_files +=1
            if(ham_prob <= spam_prob):
                misclassified+=1
    return misclassified,num_files 

print("Implementation of Naive Bayes for text classification--")
ham_test = test_path + '\ham'
spam_test = test_path + '\spam'        
misclassified_ham,tot_ham = ham_or_spam_selector(ham_test, "ham")
misclassified_spam,tot_spam = ham_or_spam_selector(spam_test,"spam")
ham_accuracy = round(((tot_ham - misclassified_ham )/(tot_ham ))*100,2)
spam_accuracy = round(((tot_spam -  misclassified_spam )/(tot_spam))*100,2)
emails_classified = tot_ham + tot_spam
tot_misclassified = misclassified_ham + misclassified_spam
accuracy = round(((emails_classified  - tot_misclassified )/emails_classified)*100,2)





print("\nNaive Bayes Accuracy For Ham Emails Classification:" + str(ham_accuracy) + "%")
print("\nNaive Bayes Accuracy For Spam Emails Classification: " + str(spam_accuracy) + "%") 
print("\nNaive Bayes Total accuracy for Test Emails: " + str(accuracy) + "% + \n")


print("Executing Naive Bayes after removing stop words")
word_probability("ham",1)
word_probability("spam",1) 

misclassified_ham,tot_ham = ham_or_spam_selector(ham_test, "ham")
misclassified_spam,tot_spam = ham_or_spam_selector(spam_test,"spam")
ham_accuracy = round(((tot_ham - misclassified_ham )/(tot_ham ))*100,2)
spam_accuracy = round(((tot_spam -  misclassified_spam )/(tot_spam))*100,2)
emails_classified = tot_ham + tot_spam
tot_misclassified = misclassified_ham + misclassified_spam
accuracy = round(((emails_classified  - tot_misclassified )/emails_classified)*100,2)


print("\nNaive Bayes Accuracy For Ham Emails Classification:" + str(ham_accuracy) + "%")
print("\nNaive Bayes Accuracy For Spam Emails Classification: " + str(spam_accuracy) + "%") 
print("\nNaive Bayes Total accuracy for Test Emails: " + str(accuracy) + "%")