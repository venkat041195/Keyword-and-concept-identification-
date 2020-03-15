# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:35:53 2019

@author: Arnaz Tantra, Navy Merianda & Venkata Ramana Pola
"""
import pandas as pd
import os
from datetime import datetime
import datetime as dt
import sqlite3
import re
import nltk
import numpy as np
#nltk.download('stopwords')
#nltk.download('punkt')
import csv
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.linear_model import LogisticRegression

#Main entry to the code. Contros the overall flow of this program.
def main():
    try:
        #All the function calls are disabled as they have to be run in a certain sequence.
        
        #CreateDatabaseFileFromTrainingSet()
        #numberOfRows = TotalRowsInTrainSet()
        #duplicatesData = CheckForDuplicates()
        #ClearDuplicatesAndCreateNewDataFile(duplicatesData)
        #print('Total Duplicates : ' + str(numberOfRows - duplicatesData.shape[0]))
        #tagsData = GetTagsData()
        #tagsDataModel = GetUniqueTagsAndTagsDictionary(tagsData)
        #ExploreAndPlotTagsData(tagsDataModel)
        #GetDataAfterPreprocessing()
        #pData = GetPreprocessedData()
        #questionsExplained = ConvertTagsToMultiOpVariables(pData)
        #Featurize(pData)
        
    except Exception as GeneralException:
        print(GeneralException)
    

#Creates the sqllite database file from training set.
def CreateDatabaseFileFromTrainingSet():
    if not os.path.isfile('train.db'):
        disk_engine = create_engine('sqlite:///train.db')
        start = dt.datetime.now()
        chunkLength = 100000
        totalCount = 0
        index_start = 1
        for dataFrame in pd.read_csv('Train.csv', names=['Id', 'Title', 'Body', 'Tags'], chunksize = chunkLength, iterator=True, encoding='utf-8', ):
            dataFrame.index += index_start
            totalCount += 1
            print('{} rows'.format(totalCount * chunkLength))
            dataFrame.to_sql('data', disk_engine, if_exists='append')
            index_start = dataFrame.index[-1] + 1
    

#Returns the total rows from the table created above.
def TotalRowsInTrainSet():
    if os.path.isfile('train.db'):
        con = sqlite3.connect('train.db')
        numberOfRows = pd.read_sql_query("""SELECT count(Id) FROM data""", con)
        print("Number of rows in the database :","\n", numberOfRows['count(Id)'].values[0])
        con.close()
        return numberOfRows.values[0]
    else:
        print("Please check if the train.db file exists!")

#Checks for duplicates using the group by clause in the above table.
def CheckForDuplicates():
    if os.path.isfile('train.db'):
        start = datetime.now()
        con = sqlite3.connect('train.db')
        duplicatesCount = pd.read_sql_query('SELECT Title, Body, Tags, COUNT(*) as Duplicate_Count FROM data GROUP BY Title, Body, Tags', con)
        con.close()
        return duplicatesCount
        print("Time taken to run this cell :", datetime.now() - start)
    else:
        print("Please check if the train.db file exists!")

#Clears up the duplicates found in the table and creates a new table with no duplicates for further processing.
def ClearDuplicatesAndCreateNewDataFile(duplicatesData):
    duplicatesData.dropna(inplace=True)
    if not os.path.isfile('trainwithoutduplicates.db'):
        diskDuplicates = create_engine("sqlite:///trainwithoutduplicates.db")
        noDuplicates = pd.DataFrame(duplicatesData, columns=['Title', 'Body', 'Tags'])
        noDuplicates.to_sql('NoDuplicateTrain', diskDuplicates)
    
#Returns all the tags found in the above table.
def GetTagsData():
    if os.path.isfile('trainwithoutduplicates.db'):
        con = sqlite3.connect('trainwithoutduplicates.db')
        tagsData = pd.read_sql_query("""SELECT Tags FROM NoDuplicateTrain""", con)
        con.close()
        tagsData.drop(tagsData.index[0], inplace=True)
        tagsData.head()
        return tagsData
    else:
        print("Please check if the trainwithoutduplicates.db file exists!")
    
#Outputs total tags and unique tags count using a vectorizer module and creates a dictionary with tag and associated count.
def GetUniqueTagsAndTagsDictionary(tagsData):
    vectorizer = CountVectorizer(tokenizer = lambda x: x.split())
    tagsDataModel = vectorizer.fit_transform(tagsData['Tags'])
    print("Total Tags :" + str(tagsDataModel.shape[0]))
    print("Unique Tags :" + str(tagsDataModel.shape[1]))
    tags = vectorizer.get_feature_names()
    
    freqs = tagsDataModel.sum(axis = 0).A1
    result = dict(zip(tags, freqs))
    if not os.path.isfile('TagsCountDictionary.csv'):
        with open('TagsCountDictionary.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in result.items():
                writer.writerow([key, value])
    return tagsDataModel

#Using the tags count dictionary, retrieve some insight into tags.
def ExploreAndPlotTagsData(tagsDataModel):
    tagsDataframe = pd.read_csv("TagsCountDictionary.csv", names=['Tags', 'Counts'])
    tagsDataframe.head()
    
    tagsDataframeSorted = tagsDataframe.sort_values(['Counts'], ascending=False)
    print(tagsDataframeSorted[0:20])
    tagsCount = tagsDataframeSorted['Counts'].values
    
    tagsAgainstQuestionsCount = tagsDataModel.sum(axis = 1).tolist()
    tagsAgainstQuestionsCount = [int(jTag) for iCount in tagsAgainstQuestionsCount for jTag in iCount]
    print ('We have total {} Data Rows.'.format(len(tagsAgainstQuestionsCount)))
    print(tagsAgainstQuestionsCount[:5])
    
    print( "Max No of Tags Per Question: %d"%max(tagsAgainstQuestionsCount))
    print( "Min No of Tags Per Question: %d"%min(tagsAgainstQuestionsCount))
    print( "Avg. No of Tags Per Question: %f"% ((sum(tagsAgainstQuestionsCount)*1.0)/len(tagsAgainstQuestionsCount)))

#Preprocess data - start point.
def PreprocessData():
    if os.path.isfile('trainwithoutduplicates.db'):
        connection = CreateConnection('trainwithoutduplicates.db')
    if connection is not None:
        allDataConn = connection.cursor()
        allDataConn.execute("SELECT Title, Body, Tags From NoDuplicateTrain LIMIT 200001;")
        PreprocessQuestionsFromBody(allDataConn)
    
    allDataConn.close()

#Regex to remove special characters.
def RemoveHtmlTags(data):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(data))
    return cleantext

#Entire preprocess data in entailed in this function.
def PreprocessQuestionsFromBody(allData):
    stopWords = set(stopwords.words('english'))
    englishStemmer = SnowballStemmer("english")
    
    questions_with_code=0
    len_pre=0
    len_post=0
    questions_proccesed = 0
    writeQuestionsConn = CreateConnection('PreprocessedQues.db')
    writeQuestions = writeQuestionsConn.cursor()
    for row in allData:
        is_code = 0
        title, question, tags = row[0], row[1], str(row[2])
        if '<code>' in question:
            questions_with_code+=1
            is_code = 1
            x = len(question)+len(title)
            len_pre+=x
            
            code = str(re.findall(r'<code>(.*?)</code>', question, flags=re.DOTALL))
            
            question=re.sub('<code>(.*?)</code>', '', question, flags=re.MULTILINE|re.DOTALL)
            question = RemoveHtmlTags(question.encode('utf-8'))
        
        title=title.encode('utf-8')
        question=str(title)+" "+str(title)+" "+str(title)+" "+question

        question=re.sub(r'[^A-Za-z0-9#+.\-]+',' ',question)
        words=word_tokenize(str(question.lower()))
        

        question=' '.join(str(englishStemmer.stem(j)) for j in words if j not in stopWords and (len(j)!=1 or j=='c'))
        
        len_post+=len(question)
        tupleData = (question,code,tags,x,len(question),is_code)
        questions_proccesed += 1
        writeQuestions.execute("INSERT INTO QuestionsProcessed(question,code,tags,words_pre,words_post,is_code) values (?,?,?,?,?,?)", tupleData)

        
    writeQuestionsConn.commit()
    writeQuestionsConn.close()

#Retrieve processed data after removing all the characters that don't support our findings.
def GetPreprocessedData():
    if os.path.isfile('PreprocessedQues.db'):
        connection = CreateConnection('PreprocessedQues.db')
    if connection is not None:
        preprocessedData = pd.read_sql_query("""SELECT question, tags FROM QuestionsProcessed""", connection)
        
    connection.commit()
    connection.close()
    
    return preprocessedData

#Convert tags to numerical array.
def ConvertTagsToMultiOpVariables(pData):
    vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary='true')
    multilabelY = vectorizer.fit_transform(pData['tags'])
    
    questionsExplained = []
    total_tags = multilabelY.shape[1]
    total_qs = pData.shape[0]
    for i in range(50, total_tags, 50):
        questionsExplained.append(np.round(((total_qs-QuestionExplained(i, pData))/total_qs)*100,3))
    
    print("with ",50,"tags we are covering ",questionsExplained[0],"% of questions")
    print("with ",500,"tags we are covering ",questionsExplained[10],"% of questions")
    print("with ",5500,"tags we are covering ",questionsExplained[100],"% of questions")
    
    #print("with ",5500,"tags we are covering ",questionsExplained[50],"% of questions")
    #print("with ",500,"tags we are covering ",questionsExplained[0],"% of questions")
    return questionsExplained

#Return specific tags.
def TagsSelected(n, pData):
    vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary='true')
    multilabelY = vectorizer.fit_transform(pData['tags'])
    t = multilabelY.sum(axis=0).tolist()[0]
    sorted_tags_i = sorted(range(len(t)), key=lambda i: t[i], reverse=True)
    multilabelYn = multilabelY[:,sorted_tags_i[:n]]
    return multilabelYn

def QuestionExplained(n, pData):
    multilabelYn = TagsSelected(n, pData)
    x= multilabelYn.sum(axis=1)
    return (np.count_nonzero(x==0))

#Function to featurize data and test performance.
def Featurize(pData):
    multilabelYX = TagsSelected(50, pData)
    trainDataSize = 160000
    Xtrain = pData.head(trainDataSize)
    Xtest = pData.tail(pData.shape[0] - 160000)
    
    Ytrain = multilabelYX[0:trainDataSize,:]
    Ytest = multilabelYX[trainDataSize:pData.shape[0],:]
    
    print("Number of data rows in training data :", Ytrain.shape)
    print("Number of data rows in testing data :", Ytest.shape)
    
    #Tf-idf Vectorizer
    vectorizer = TfidfVectorizer(min_df=0.00009, max_features=40000, smooth_idf=True, norm="l2", \
                             tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,2))
    
    #BOW Vectorizer
    #vectorizer = CountVectorizer(min_df=0.00009, max_features=40000, ngram_range=(1,2))
    

    XTrainMultilabel = vectorizer.fit_transform(Xtrain['question'])
    XTestMultilabel = vectorizer.transform(Xtest['question'])

    classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.00001, penalty='l1'), n_jobs=-1)
    classifier.fit(XTrainMultilabel, Ytrain)
    predictions = classifier.predict (XTestMultilabel)
    
    print("Confusion Matrix")
    errorResults = multilabel_confusion_matrix(Ytest, predictions)
    
    print(errorResults)
    
    print("Accuracy :",metrics.accuracy_score(Ytest, predictions))
    print("Hamming loss ",metrics.hamming_loss(Ytest,predictions))


    precisionMicroScore = precision_score(Ytest, predictions, average='micro')
    recallMicroScore = recall_score(Ytest, predictions, average='micro')
    f1MicroScore = f1_score(Ytest, predictions, average='micro')
    
    print("Micro Average")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precisionMicroScore, recallMicroScore, f1MicroScore))

    precisionMacroScore = precision_score(Ytest, predictions, average='macro')
    recallMacroScore = recall_score(Ytest, predictions, average='macro')
    f1MacroScore = f1_score(Ytest, predictions, average='macro')
    
    print("Macro Average")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precisionMacroScore, recallMacroScore, f1MacroScore))

 
#Create database connection.   
def CreateConnection(databaseFile):
    try:
        conn = sqlite3.connect(databaseFile)
        return conn
    except Exception as ConnError:
        print(ConnError)
    return None

main()
