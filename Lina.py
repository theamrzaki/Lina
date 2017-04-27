from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv

import timeit
import pickle
import random

from stat_parser import Parser, display_tree
#from nltk.chunk import ne_chunk
#from nltk.tag import pos_tag
#from nltk.tokenize import word_tokenize

import nltk

#_____Fact Questions Libraries_____
import re
from bs4 import BeautifulSoup
import requests





#-------------------------TF-IDF cosine similarity for intnents--------------------------------#
def intents(intnent_test_sentence):

    intents_sentences=[]
    intents_sentences.append("intent")#just to adjuxt index
    test_set = (intnent_test_sentence,"")

    try:
         #--------------to use----------------------#
         f = open('tfidf_vectorizer_intent.pickle', 'rb')
         tfidf_vectorizer = pickle.load(f)
         f.close()

         f = open('tfidf_matrix_train_intent.pickle', 'rb')
         tfidf_matrix_train = pickle.load(f)
         f.close()
         #-----------------------------------------#
    except:
        # ---------------to train------------------#
        
         ENGLISH_STOP_WORDS = frozenset([
            "a", "about", "above", "across", "after", "afterwards", "again", "against",
            "all", "almost", "alone", "along", "already", "also", "although", "always",
            "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
            "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
            "around", "as", "at", "back", "be", "became", "because", "become",
            "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
            "below", "beside", "besides", "between", "beyond", "bill", "both",
            "bottom", "but", "by", "can", "cannot", "cant", "co", "con",
            "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
            "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
            "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
            "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
            "find", "fire", "first", "five", "for", "former", "formerly", "forty",
            "found", "four", "from", "front", "full", "further", "get", "give", "go",
            "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
            "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
            "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
            "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
            "latterly", "least", "less", "ltd", "made", "many", "may", "me",
            "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
            "move", "much", "must", "my", "myself", "name", "namely", "neither",
            "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
            "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
            "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
            "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
            "please", "put", "rather", "re", "same", "see", "seem", "seemed",
            "seeming", "seems", "serious", "several", "she", "should", "show", "side",
            "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
            "something", "sometime", "sometimes", "somewhere", "still", "such",
            "system", "take", "ten", "than", "that", "the", "their", "them",
            "themselves", "then", "thence", "there", "thereafter", "thereby",
            "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
            "third", "this", "those", "though", "three", "through", "throughout",
            "thru", "thus", "to", "together", "too", "top", "toward", "towards",
            "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
            "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
            "whence", "whenever", "where", "whereafter", "whereas", "whereby",
            "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
            "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
            "within", "without", "would", "yet", "you", "your", "yours", "yourself",
            "yourselves"])
    
         i=0
         with open("intents - Copy.csv", "r") as sentences_file:
             reader = csv.reader(sentences_file, delimiter=',')
             for row in reader:
                 a= row[0]
                 intents_sentences.append(row[0])
                 i+=1
    
         tfidf_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words=ENGLISH_STOP_WORDS)#stop_words='english'
         tfidf_matrix_train = tfidf_vectorizer.fit_transform(intents_sentences)  #finds the tfidf score with normalization
    
         f = open('tfidf_vectorizer_intent.pickle', 'wb')
         pickle.dump(tfidf_vectorizer, f)
         f.close()

         f = open('tfidf_matrix_train_intent.pickle', 'wb')
         pickle.dump(tfidf_matrix_train, f)
         f.close()
         # ----------------------------------------#


    tfidf_matrix_test =tfidf_vectorizer.transform(test_set)
    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)  

    cosine = np.delete(cosine,0)
    max = cosine.max()
    if max==0:
        return "normal setance","normal setance" ,"normal setance" , "100% sure"
    response_index = np.where(cosine == max)[0][0] +1 # no offset at all +3
    

    j=0
    with open("intents - Copy.csv", "r") as sentences_file:
       reader = csv.reader(sentences_file, delimiter=',')
       for row in reader:
           j += 1 # we begin with 1 not 0 &    j is initialized by 0
           if j == response_index: 
               if max<0.7:
                  return "normal setance","normal setance" ,"normal setance" , str(max)
                  # return row[1],row[2] ,"not sure" , str(max)
               else:
                  return row[1],row[2]  ,"sure" , str(max)
               break

#-------------------------Parse and see if it is for internet----------------------------------#

tree_output = []
imp_list_array={ 'Noun': [] }

def traverse(parent , x):
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == 'ROOT':
                # "======== Sentence ========="
                #print "Sentence:", " ".join(node.leaves()) , " +  type " , node.label()
                a=6
            else:
                 element_type  = node.label()
                 element_value = node.leaves()[0]
                 element_sentence = node.leaves()
                 
                 if str(element_type) =='NN'  or str(element_type) =='NNS' or str(element_type) =='NNP' or str(element_type) =='NNPS':
                    imp_list_array['Noun'].append(str(element_value))

                 #tree_output.append(node)

                 traverse(node,x)
        else:
             #tree_output.append(  node)
             tree_output.append(parent.label())

             #print "Word:", node
             a=5

def parse(sentenace):
    parser = Parser()
    tree = parser.parse(sentenace) 

    #display_tree(tree)

    for i in range(len(tree)):
        traverse(tree[i] , 0)

    tree_output_str=""
    for a in tree_output  :
         tree_output_str +=" - " + a
    print  tree_output_str
    special_parses=[
    "WRB - JJ - NNS"                   ,#   how many Leopards
    "WRB - JJ - JJ"                    ,#   how many leopards
    "WRB - JJ - VBP - DT - NN"         ,#   how big are the pyramids
    "WRB - JJ - VBZ - JJ"              ,#   how old is obama
    "WRB - JJ - VBZ - NNP"             ,#   how old is Obama
    "WRB - JJ - NN - VBP - NNP - VBP"  ,#   how much money do Bill have 

    "WP - VBD - DT - NN"               ,#   who won the champions last week        #when was the tv first invented
    "WP - VBD - NN"                    ,#   who worked today
                          
    "WP - VBP - DT - NN"               ,#   what are the pyramids
    "WP - VBZ - DT - NN - IN - NN"     ,#   what is the capital of egypt

    "WDT - NNS"                        ,#   which companies are the biggest ,
     "WRB - VBZ - NN"                   , #where is egypt
     "WP - VBZ - NNP"                    ,#what is Egypt
     "WP - VBZ - JJ"                      #what is egypt
     ]

    
    try:
        regex ="WP - VBP - PRP - VB - IN - NN"    #what do you know about egypt
        pos_tree_output = tree_output_str.index(re.search(regex, tree_output_str).group(0))
        pos_var = len(tree_output_str.replace('-', '').split()) - len(
        tree_output_str[pos_tree_output:].replace('-', '').split())
        fact_question= ' '.join(var.split()[pos_var:])

        base_address = "http://api.duckduckgo.com/?q="+imp_list_array["Noun"][0]+"&format=xml"
    
        super_page  = requests.get(base_address)
        soup_super_page = BeautifulSoup(super_page.content, "xml")
    
        answer=soup_super_page.findAll('Abstract')[0].text
        if (answer==""):
             answer=soup_super_page.findAll('Text')[0].text
        return True,answer

    except:
        try:
             #other special parses
            regex = reduce(lambda x, y: x + "|" + y, special_parses)
            pos_tree_output = tree_output_str.index(re.search(regex, tree_output_str).group(0))
            pos_var = len(tree_output_str.replace('-', '').split()) - len(
            tree_output_str[pos_tree_output:].replace('-', '').split())
            fact_question= ' '.join(var.split()[pos_var:])
    
            base_address = "http://api.duckduckgo.com/?q="+fact_question+"&format=xml"
    
            super_page  = requests.get(base_address)
            soup_super_page = BeautifulSoup(super_page.content, "xml")
    
            answer=soup_super_page.findAll('Abstract')[0].text
            if (answer==""):
                 answer=soup_super_page.findAll('Text')[0].text
            return True,answer
        except:
             return False,""

#-------------------------General DataSet--------------------------------#
def talk_to_lina(test_set_sentance):
    i=0
    sentences=[]

    #enter your test sentance 
    test_set = (test_set_sentance,"")


    #malaksh da3awa beeh
    #sentences.append(" No you.")

    #tell which sentance to stop training at
    #stop_at_sentence =20


    




    import pickle

    try:
        ##--------------to use------------------#
         f = open('tfidf_vectorizer_april.pickle', 'rb')
         tfidf_vectorizer = pickle.load(f)
         f.close()
         
         f = open('tfidf_matrix_train_april.pickle', 'rb')
         tfidf_matrix_train = pickle.load(f)
         f.close()
        #----------------------------------------#
    except:
        #---------------to train------------------#
        start = timeit.default_timer()
    
         #enter jabberwakky sentance
        with open("Lina_all.csv", "r") as sentences_file:
             reader = csv.reader(sentences_file, delimiter=',')
             #reader.next()
             #reader.next()
             for row in reader:
                 #if i==stop_at_sentence:
                 #    break
                 sentences.append(row[0])
                 i+=1

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(sentences)  #finds the tfidf score with normalization
        #tfidf_matrix_test =tfidf_vectorizer.transform(test_set)
        stop = timeit.default_timer()
        print ("training time took was : ")
        print stop - start 

        f = open('tfidf_vectorizer_april.pickle', 'wb')
        pickle.dump(tfidf_vectorizer, f)
        f.close()
        
        f = open('tfidf_matrix_train_april.pickle', 'wb')
        pickle.dump(tfidf_matrix_train, f)
        f.close()
        #-----------------------------------------#


    tfidf_matrix_test =tfidf_vectorizer.transform(test_set)

    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)  


    cosine = np.delete(cosine,0)
    max = cosine.max()
    response_index =0
    if(max>0.7):
         new_max = max-0.05
         list = np.where(cosine > new_max)
         print ("number of responses with 0.05 from max = " + str(list[0].size) )
         response_index = random.choice(list[0])

    else:
        print ("not sure")
        print ("max is = " + str(max) )
        response_index = np.where(cosine == max)[0][0] +2# no offset at all +3

    j=0
    with open("Lina_all.csv", "r") as sentences_file:
       reader = csv.reader(sentences_file, delimiter=',')
       for row in reader:
           j += 1 # we begin with 1 not 0 &    j is initialized by 0
           if j == response_index: 
               return row[1]
               break
   
#-------------------------------------------------------------------------#



while True:
    var = raw_input("Talk to Lina: ")
    #var ="where is egypt"
    result = intents(var)

    if(result[1] =="normal setance"):
        fact_question = parse(var)  
        if(fact_question[0]):
            print "Fact Question"
            print fact_question[1]

        else:
            print "action : "  +  result[0]
            print "Lina : "    +  talk_to_lina(var)
            print



    else:# can be an intent
                        #if(result[2]=="not sure"):
    #    #    option = raw_input(random.choice(pre_offer_varaibles) + random.choice(offer_varaibles) + result[0] +" ? ")

    #    #    if any(word in option for word in yes_variables):
    #    #        print "action : "           +  result[0]
    #    #        print "class : "            +  result[1]
    #    #        print "certainty : "        +  result[2]
    #    #        print "certainty level : "  +  result[3]
    #    #        print 
    #    #    else:
    #    #        print "Lina : "    +  talk_to_lina(var)
    #    #        print

        #else:#sure intent
        print "action : "           +  result[0]
        print "class : "            +  result[1]
        print "certainty : "        +  result[2]
        print "certainty level : "  +  result[3]
        print 
        
    tree_output = []
    imp_list_array["Noun"]=[]
    tree_output_str = ""
















#----------naive bayes classifier intents       ----------------

#import pandas
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.feature_extraction.text import CountVectorizer

#def intents(sentance):
#    #--------------------to learn------------------------------#
#    #intent = pandas.read_csv("intents - Copy.csv")
#    #vectorizer = CountVectorizer()
#    #features = vectorizer.fit_transform(intent.iloc[:,0])
#    #model = MultinomialNB()
#    #model.fit(features, intent.iloc[:,2])

#    #f = open('CountVectorizer.pickle', 'wb')
#    #pickle.dump(vectorizer, f)
#    #f.close()

#    #f = open('MultinomialNB.pickle', 'wb')
#    #pickle.dump(model, f)
#    #f.close()

#    #------------------ to use --------------------------------#
#    f = open('CountVectorizer.pickle', 'rb')
#    vectorizer = pickle.load(f)
#    f.close()

#    f = open('MultinomialNB.pickle', 'rb')
#    model = pickle.load(f)
#    f.close()

#    test_feature = vectorizer.transform((sentance,));
#    pred_prob = max(model.predict_proba(test_feature)[0])
#    pred = model.predict(test_feature)[0]

#    return pred, pred_prob

#while True:
#    var = raw_input("Talk to Lina: ")
#    result = intents(var)

#    if (result[1] > 0.5):#sure to be intent
#        print "Command : " +result[0], result[1]
#        print

#    else:
#        print "Lina : " +  talk_to_lina(var)
#        print


#----------named entity      ----------------
#def get_named_entities(sentenace):
#  ne_tree = ne_chunk(pos_tag(word_tokenize(sentenace)))
#  named_entities=[]

#  for t in ne_tree:
#     try:
#         if t.label() is not None:
#             named_entities.append( {'Entty':t.label() , 'Name':t.leaves()[0][0] , 'Type' : t.leaves()[0][1] } )
#     except:
#       aaqq =5

#  return named_entities

#result_parse = parse(var)  
#named_entities = get_named_entities(var)
#if(result_parse[0] is True and  len(named_entities) is not 0):
#    print ("search internet for " )
#    print named_entities
#    print
#    continue



#var = raw_input("Talk to Lina: ")
#print "Lina : " +  talk_to_lina(var)


#-------------Lina asks for clrification ----------------
#yes_variables= [
#'yes'         ,
#'yea'         ,
#'OK'          ,
#'okey-dokey'  ,
#'by all means',
#'affirmative' ,
#'aye'         ,
#'roger'       ,
#'uh-huh'      ,
#'right'       ,
#'very well'   ,
#'yup'         ,
#'yuppers'     ,
#'right on'    ,
#'surely'      ,
#'totally'     ,
#'sure'        ,
#]
#pre_offer_varaibles=[
#' I\'m not quite sure I know what you mean.  ',
#' I\'m not quite sure I follow you.',
#' I don\'t quite see what you mean.',
#' I\'m not sure I got your point.',
#' Sorry, I didn\'t quite hear what you said ',
#' Sorry, I didn\'t get your point.',
#' I don\'t quite see what you\'re getting at. ']
#offer_varaibles=['would you like me to ',
#'do you want me to ' ,
#'did you mean to ']