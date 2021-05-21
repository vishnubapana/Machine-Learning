import collections
import os
import re
import codecs
import numpy

#train = input("Enter the path for the training data: ")
#test = input("Enter the path for the testing data: ")
#stopword_bool= input("Should stopwords be removed(enter 'yes' or 'no'): ").lower()
#lambda_var = input("Enter a lambda value: ")
#itera = input("Enter the number of desired iteras: ")


def read_stopwords():
    if (os.path.isfile("stopwords.txt")) != True:
        print("Stopwords file not found!")
    else: 
        with open('stopwords.txt') as f:
            file = f.read().lower().replace('\n', ' ').split(' ')
        return file


def file_function(filename, path):
    handler = codecs.open(path + "\\" + filename, 'rU','latin-1')
    words = [locate.lower() for locate in re.findall('[A-Za-z0-9\']+', handler.read())]
    handler.close()
    return words


def file_finder(path):
    word_list = list()
    fileCount = 0
    for files in os.listdir(path):
        if files.endswith(".txt"):
            word_list = word_list + file_function(files, path)
            fileCount = fileCount + 1
    return word_list, fileCount


def delete_stopwords():
    for word in stopWords:
        if word in ham:
            w = 0
            ham_len=len(ham)
            while (w < ham_len):
                if (ham[w] == word):
                    ham.remove(word)
                    ham_len = ham_len - 1
                    continue
                w = w + 1
        if word in spam:
            x = 0
            spam_len=len(spam)
            while (x < spam_len):
                if (spam[x] == word):
                    spam.remove(word)
                    spam_len = spam_len - 1
                    continue
                x = x + 1
        if word in ham_test:
            y = 0
            ham_len_test=len(ham_test)
            while (y < ham_len_test):
                if (ham_test[y] == word):
                    ham_test.remove(word)
                    ham_len_test = ham_len_test - 1
                    continue
                y = y + 1
        if word in spam_test:
            z = 0
            spam_len_test=len(spam_test)
            while (z < spam_len_test):
                if (spam_test[z] == word):
                    spam_test.remove(word)
                    spam_len_test = spam_len_test - 1
                    continue
                z = z + 1


def create_mat(row, column):
    featureMatrix = [0] * row
    for i in range(row):
        featureMatrix[i] = [0] * column
    return featureMatrix


def build_mat(featureMatrix, path, listbog, rmat, classifier, fin_list):
    for fileName in os.listdir(path):
        words = file_function(fileName, path)
        temp = dict(collections.Counter(words))
        for key in temp:
            if key in listbog:
                column = listbog.index(key)
                featureMatrix[rmat][column] = temp[key]
        if (classifier == "ham"):
            fin_list[rmat] = 0
        elif (classifier == "spam"):
            fin_list[rmat] = 1
        rmat = rmat + 1
    return featureMatrix, rmat, fin_list



def sigmoid(val):
    return (1 / (1 + numpy.exp(-val)))
    

def sig_funct(totalFiles, totalFeatures, featureMatrix):
    global sigMoidList
    for files in range(totalFiles):
        summation = 1.0

        for features in range(totalFeatures):
            summation = summation + featureMatrix[files][features] * weightOfFeature[features]
        sigMoidList[files] = sigmoid(summation)


def calc_weight(totalFiles, numberOfFeature, featureMatrix, fin_list):
    global sigMoidList

    for feature in range(numberOfFeature):
        weight = 0
        for files in range(totalFiles):
            frequency = featureMatrix[files][feature]
            y = fin_list[files]
            sigmoidValue = sigMoidList[files]
            weight = weight + (frequency * (y - sigmoidValue))

        oldW = weightOfFeature[feature]
        weightOfFeature[feature] = weightOfFeature[feature] + ((weight * rate) - (rate * lambda_var * oldW))

    return weightOfFeature


def train_func(totalFiles, numbeOffeatures, train_mat, fin_list):
    sig_funct(totalFiles, numbeOffeatures, train_mat)
    calc_weight(totalFiles, numbeOffeatures, train_mat, fin_list)


def classification_func():
    cham, wham, cspam, wspam, i = 0, 0, 0, 0, 0
    for file in range(totalTestFiles):
        summation = 1.0
        for i in range(len(testListbog)):
            word = testListbog[i]

            if word in listbog:
                index = listbog.index(word)
                weight = weightOfFeature[index]
                wordcount = test_mat[file][i]

                summation = summation + (weight * wordcount)

        sigSum = sigmoid(summation)
        if (testfin_list[file] == 0):
            if sigSum < 0.5:
                cham = cham + 1.0
            else:
                wham = wham + 1.0
        else:
            if sigSum >= 0.5:
                cspam = cspam + 1.0
            else:
                wspam = wspam + 1.0
        i = i + 1
    tot_ham = cham + wham
    tot_spam = cspam + wspam
    total = tot_ham + tot_spam
    print("Ham Accuracy:" + str((cham / tot_ham) * 100))
    print("Spam Accuracy:" + str((cspam / tot_spam) * 100))
    print("Accuracy:" + str(((cham+cspam) / total) * 100))




train_path = "assignment3_train/train"
test_path = "assignment3_test/test"
stopword_bool= 'yes'
lambda_var = 100
itera = 25

ham = list()
spam = list()
ham_count = 0
spam_count = 0
rate = 0.001

stopWords = read_stopwords()


ham_folder = train_path + '/ham'
spam_folder = train_path + '/spam'
test_ham_folder = test_path + '/ham'
spam_test_folder = test_path + '/spam'


re.compile(r'[A-Za-z0-9\']')

ham, ham_count = file_finder(ham_folder)
spam, spam_count = file_finder(spam_folder)


ham_test, counttest_ham_folder = file_finder(test_ham_folder)
spam_test, countspam_test_folder = file_finder(spam_test_folder)

if (stopword_bool == "yes"):
    delete_stopwords()


bog = ham + spam
dictbog = collections.Counter(bog)
listbog = list(dictbog.keys())
fin_list = list() 
totalFiles = ham_count + spam_count


testbog = ham_test + spam_test
testDictbog = collections.Counter(testbog)
testListbog = list(testDictbog.keys())
testfin_list = list() 
totalTestFiles = counttest_ham_folder + countspam_test_folder
train_mat = create_mat(totalFiles, len(listbog))
test_mat = create_mat(totalTestFiles, len(testListbog))

rmat = 0
testrmat = 0

sigMoidList = list()
for i in range(totalFiles):
    sigMoidList.append(-1)
    fin_list.append(-1)

for i in range(totalTestFiles):
    testfin_list.append(-1)

weightOfFeature = list()

for feature in range(len(listbog)):
    weightOfFeature.append(0)



train_mat, rmat, fin_list = build_mat(train_mat, ham_folder, listbog, rmat, "ham", fin_list)
train_mat, rmat, fin_list = build_mat(train_mat, spam_folder, listbog, rmat, "spam", fin_list)
test_mat, testrmat, testfin_list = build_mat(test_mat, test_ham_folder, testListbog, testrmat, "ham", testfin_list)
test_mat, testrmat, testfin_list = build_mat(test_mat, spam_test_folder, testListbog, testrmat, "spam", testfin_list)


print("Training, Please wait...")
for i in range(int(itera)):
    train_func(totalFiles, len(listbog), train_mat, fin_list)


print("Algorithm is trained, test data will now be classififed.")
classification_func()