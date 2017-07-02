from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv
import os

import timeit
import pickle
import random

########karim commented this########
from stat_parser.parser import Parser, display_tree
####################################
# from nltk.chunk import ne_chunk
# from nltk.tag import pos_tag
# from nltk.tokenize import word_tokenize

import nltk
from extract_intents import extract_intents

# _____Fact Questions Libraries_____
import re
from bs4 import BeautifulSoup
import requests

# _____Curse Fiter_____
import filter


#______Intent classifier Youssef_____
import pandas
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import numpy as np


# -----------------------------------$$ Global Variables $$-------------------------------------#
delimeter = "_+^$#*#$^+_"
dir = os.path.dirname(__file__)


# -------------------------TF-IDF cosine similarity for intnents--------------------------------#
#old intent classifier
#def intents(intent_test_sentence):
#    intents_sentences = []
#    intents_sentences.append("intent")  # just to adjuxt index
#    test_set = (intent_test_sentence, "")
#    tfidf_vectorizer_intent_path = os.path.join(dir, 'tfidf_vectorizer_intent.pickle')
#    tfidf_matrix_train_intent_path = os.path.join(dir, 'tfidf_matrix_train_intent.pickle')
#    intents_copy_path = os.path.join(dir, "intents - Copy.csv")

#    try:
#        # --------------to use----------------------#
#        f = open(tfidf_vectorizer_intent_path, 'rb')
#        tfidf_vectorizer = pickle.load(f)
#        f.close()

#        f = open(tfidf_matrix_train_intent_path, 'rb')
#        tfidf_matrix_train = pickle.load(f)
#        f.close()
#        # -----------------------------------------#
#    except:
#        # ---------------to train------------------#

#        ENGLISH_STOP_WORDS = frozenset([
#            "a", "about", "above", "across", "after", "afterwards", "again", "against",
#            "all", "almost", "alone", "along", "already", "also", "although", "always",
#            "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
#            "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
#            "around", "as", "at", "back", "be", "became", "because", "become",
#            "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
#            "below", "beside", "besides", "between", "beyond", "bill", "both",
#            "bottom", "but", "by", "can", "cannot", "cant", "co", "con",
#            "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
#            "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
#            "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
#            "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
#            "find", "fire", "first", "five", "for", "former", "formerly", "forty",
#            "found", "four", "from", "front", "full", "further", "get", "give", "go",
#            "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
#            "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
#            "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
#            "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
#            "latterly", "least", "less", "ltd", "made", "many", "may", "me",
#            "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
#            "move", "much", "must", "my", "myself", "name", "namely", "neither",
#            "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
#            "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
#            "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
#            "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
#            "please", "put", "rather", "re", "same", "see", "seem", "seemed",
#            "seeming", "seems", "serious", "several", "she", "should", "show", "side",
#            "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
#            "something", "sometime", "sometimes", "somewhere", "still", "such",
#            "system", "take", "ten", "than", "that", "the", "their", "them",
#            "themselves", "then", "thence", "there", "thereafter", "thereby",
#            "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
#            "third", "this", "those", "though", "three", "through", "throughout",
#            "thru", "thus", "to", "together", "too", "top", "toward", "towards",
#            "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
#            "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
#            "whence", "whenever", "where", "whereafter", "whereas", "whereby",
#            "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
#            "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
#            "within", "without", "would", "yet", "you", "your", "yours", "yourself",
#            "yourselves"])

#        i = 0
#        with open(intents_copy_path, "r") as sentences_file:
#            reader = csv.reader(sentences_file, delimiter=',')
#            for row in reader:
#                a = row[0]
#                intents_sentences.append(row[0])
#                i += 1

#        tfidf_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2),
#                                           stop_words=ENGLISH_STOP_WORDS)  # stop_words='english'
#        tfidf_matrix_train = tfidf_vectorizer.fit_transform(
#            intents_sentences)  # finds the tfidf score with normalization

#        f = open(tfidf_vectorizer_intent_path, 'wb')
#        pickle.dump(tfidf_vectorizer, f)
#        f.close()

#        f = open(tfidf_matrix_train_intent_path, 'wb')
#        pickle.dump(tfidf_matrix_train, f)
#        f.close()
#        # ----------------------------------------#

#    tfidf_matrix_test = tfidf_vectorizer.transform(test_set)
#    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)

#    cosine = np.delete(cosine, 0)
#    max = cosine.max()
#    if max == 0:
#        return "normal sentence", "normal sentence", "normal sentence", "100% sure"
#    response_index = np.where(cosine == max)[0][0] + 1  # no offset at all +3

#    j = 0
#    with open(intents_copy_path, "r") as sentences_file:
#        reader = csv.reader(sentences_file, delimiter=',')
#        for row in reader:
#            j += 1  # we begin with 1 not 0 &    j is initialized by 0
#            if j == response_index:
#                if max < 0.7:
#                    return "normal sentence", "normal sentence", "normal sentence", str(max)
#                    # return row[1],row[2] ,"not sure" , str(max)
#                else:
#                    return row[1], row[2], "sure", str(max)
#                break

#Youssef's intent classifier
def _getFormattedPlainText(datainput):
    text = ""
    for data in datainput:
        text += str(data) + "\n"
    text = text.lower()
    text = text.translate(None, string.punctuation)
    return text

#return intent type, action
def getAnswer(inputString, threshold):
    data = pandas.read_csv("intents.csv")

    inputString = inputString.lower()
    inputString = inputString.translate(None, string.punctuation)

    trainDocs = _getFormattedPlainText(data.Input)
    trainDocsSplitted = trainDocs.split('\n')

    alltext = trainDocs + "\n" + inputString

    globalVector = TfidfVectorizer(lowercase=True, stop_words='english', analyzer='word')
    tfidf_vec = globalVector.fit_transform(alltext.split('\n'))
    pairwise_similarity = (tfidf_vec * tfidf_vec.T).A

    testSim = pairwise_similarity[len(pairwise_similarity) - 1]
    testSim[len(trainDocsSplitted)] = 0

    maxvalue = testSim.max()
    index = np.where(testSim == maxvalue)[0]
    if(maxvalue < threshold):
        return (None, None)
    else:
        _index = np.random.choice(index)
        return(data.IntentType[_index], data.Action[_index])




# -------------------------Parse and see if it is for internet----------------------------------#

tree_output = []
imp_list_array = {'Noun': []}


def traverse(parent, x):
    try:
        for node in parent:
            if type(node) is nltk.Tree:
                if node.label() == 'ROOT':
                    # "======== Sentence ========="
                    # print "Sentence:", " ".join(node.leaves()) , " +  type " , node.label()
                    a = 6
                else:
                    element_type = node.label()
                    element_value = node.leaves()[0]
                    element_sentence = node.leaves()

                    if str(element_type) == 'NN' or str(element_type) == 'NNS' or str(element_type) == 'NNP' or str(
                            element_type) == 'NNPS':
                        imp_list_array['Noun'].append(str(element_value))

                    # tree_output.append(node)

                    traverse(node, x)
            else:
                # tree_output.append(  node)
                tree_output.append(parent.label())

                # print "Word:", node
                a = 5
    except:
        tree_output.append('NN')


def parse(sentence):
    while len(tree_output) > 0:
        tree_output.pop()

    parser = Parser()
    try:
        tree = parser.parse(sentence)
        print tree
    except:
        return False, ""

    # display_tree(tree)
    print("parse succeeded")

    for i in range(len(tree)):
        traverse(tree[i], 0)

    print("traverse succeeded")
    tree_output_str = ""

    for a in tree_output:
        tree_output_str += " - " + a
    print  tree_output_str
    special_parses = [
        "WRB - JJ - NNS",  # how many Leopards
        "WRB - JJ - JJ",  # how many leopards
        "WRB - JJ - VBP - DT - NN",  # how big are the pyramids
        "WRB - JJ - VBZ - JJ",  # how old is obama
        "WRB - JJ - VBZ - NNP",  # how old is Obama
        "WRB - JJ - NN - VBP - NNP - VBP",  # how much money do Bill have
        "WRB - VBP - DT - NN",  # where are the pyramids
        "WP - VBP - PRP - VB - IN - NN",  # what do you know about egypt

        "WP - VBD - DT - NN",  # who won the champions last week        #when was the tv first invented
        "WP - VBD - NN",  # who worked today

        "WP - VBP - DT - NN",  # what are the pyramids
        "WP - VBZ - DT - NN - IN - NN",  # what is the capital of egypt

        "WDT - NNS",  # which companies are the biggest ,
        "WRB - VBZ - NN",  # where is egypt
        "WP - VBZ - NNP",  # what is Egypt
        "WP - VBZ - JJ",  # what is egypt
        "WRB - VBD - NNP",  # when did Bayern
        "WP - VBZ - NN"  # what is indonesian
    ]

    try:
        # other special parses
        regex = reduce(lambda x, y: x + "|" + y, special_parses)
        print tree_output_str
        pos_tree_output = tree_output_str.index(re.search(regex, tree_output_str).group(0))
        pos_var = len(tree_output_str.replace('-', '').split()) - len(
            tree_output_str[pos_tree_output:].replace('-', '').split())
        print pos_var
        print tree_output_str
        fact_question = ' '.join(sentence.split()[pos_var:])

        print("it is a fact question")
        base_address = "https://api.duckduckgo.com/?q=" + fact_question + "&format=xml"

        super_page = requests.get(base_address)
        print("request succeeded")

        soup_super_page = BeautifulSoup(super_page.content, "xml")
        print("BeautifulSoup succeeded")

        answer = soup_super_page.findAll('Abstract')[0].text
        if (answer == ""):
            answer = soup_super_page.findAll('Text')[0].text
        return True, answer
    except Exception as exception:
        print ("error2", exception)
        print (type(exception).__name__)
        print (exception.__class__.__name__)
        return False, ""  # -----------------------General DataSet   &   Movies Lines----------------#


def talk_to_lina(test_set_sentence, csv_file_path, tfidf_vectorizer_pikle_path, tfidf_matrix_train_pikle_path):
    i = 0
    sentences = []

    # enter your test sentence
    test_set = (test_set_sentence, "")

    # 3ashan yzabt el indexes
    sentences.append(" No you.")
    sentences.append(" No you.")

    try:
        ##--------------to use------------------#
        f = open(tfidf_vectorizer_pikle_path, 'rb')
        tfidf_vectorizer = pickle.load(f)
        f.close()

        f = open(tfidf_matrix_train_pikle_path, 'rb')
        tfidf_matrix_train = pickle.load(f)
        f.close()
        # ----------------------------------------#
    except:
        # ---------------to train------------------#
        start = timeit.default_timer()

        # enter jabberwakky sentence
        with open(csv_file_path, "r") as sentences_file:
            reader = csv.reader(sentences_file, delimiter=',')
            # reader.next()
            # reader.next()
            for row in reader:
                # if i==stop_at_sentence:
                #    break
                sentences.append(row[0])
                i += 1

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(sentences)  # finds the tfidf score with normalization
        # tfidf_matrix_test =tfidf_vectorizer.transform(test_set)
        stop = timeit.default_timer()
        print ("training time took was : ")
        print stop - start

        f = open(tfidf_vectorizer_pikle_path, 'wb')
        pickle.dump(tfidf_vectorizer, f)
        f.close()

        f = open(tfidf_matrix_train_pikle_path, 'wb')
        pickle.dump(tfidf_matrix_train, f)
        f.close()
        # -----------------------------------------#

    tfidf_matrix_test = tfidf_vectorizer.transform(test_set)

    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)

    cosine = np.delete(cosine, 0)
    max = cosine.max()
    response_index = 0
    if (max > 0.7):
        new_max = max - 0.01
        list = np.where(cosine > new_max)
        print ("number of responses with 0.01 from max = " + str(list[0].size))
        response_index = random.choice(list[0])

    else:
        print ("not sure")
        print ("max is = " + str(max))
        response_index = np.where(cosine == max)[0][0] + 2  # no offset at all +3

    j = 0

    with open(csv_file_path, "r") as sentences_file:
        reader = csv.reader(sentences_file, delimiter=',')
        for row in reader:
            j += 1  # we begin with 1 not 0 &    j is initialized by 0
            if j == response_index:

                if delimeter in row[1]:
                    # get newest suggestion
                    answer_row = row[1].split(delimeter)
                    row[1] = answer_row[1]

                else:  # add new suggestion
                    note = "just return old original suggestion"

                return row[1], response_index,
                break


# -------------------------------------------------------------------------#

# -----------------------Edit Module (RealTime Learn)----------------------#
def edit_real_time(new_sentence,dataset_number, LineID):
    dataset_path = ["Lina_all.csv",
                    "Conversations/action_conversation.csv",
                    "Conversations/animation_conversation.csv",
                    "Conversations/comedy_conversation.csv",
                    "Conversations/crime_conversation.csv",
                    "Conversations/drama_conversation.csv",
                    "Conversations/fantasy_conversation.csv",
                    "Conversations/film-noir.csv_conversation.csv",
                    "Conversations/horror_conversation.csv",
                    "Conversations/romance_conversation.csv",
                    "Conversations/sci-fi_conversation.csv",
                    "Conversations/war_conversation.csv"]
    print
    if filter.curse_no_marks(new_sentence):
        try:
            f = open(dataset_path[dataset_number], 'r')
            reader = csv.reader(f)
            mylist = list(reader)
            f.close()

            if delimeter in mylist[LineID - 1][1]:
                # discard old suggestion
                answer_row = mylist[LineID - 1][1].split(delimeter)
                mylist[LineID - 1][1] = answer_row[0] + delimeter + new_sentence

            else:  # add new suggestion
                mylist[LineID - 1][1] += delimeter + new_sentence

            my_new_list = open(dataset_path[dataset_number], 'wb')
            csv_writer = csv.writer(my_new_list)
            csv_writer.writerows(mylist)
            my_new_list.close()
            print ("thanks for your support")
            print
        except:
            pass
    else:
        print ("contains swear word")


# -------------------------------------------------------------------------#


def callBot(var, option):
    result = extract_intents(var)
    response = ""
    if (result[1] == "normal sentence"):    #not anwar intent

        (intentType, action) = getAnswer(var, 0.7)

        if intentType == None:      #not youssef intent
            fact_question = parse(var)  # [False]
            if (fact_question[0]):
                print "Fact Question"
                # print fact_question[1].encode('utf-8')
                response = fact_question[1].encode('utf-8')
                print

            else:
                print "action : " + result[0]
                print ("ENTER CHARACTER:")
                print (
                    "general:0   action:1   animation:2   comedy:3   crime:4  drama:5   fantasy:6    filmnoir:7   horror:8  romance:9   scifi:10   war:11")
                # option = int(raw_input("enter option as number: ")   )

                if option == 0:
                    Lina_all_path = os.path.join(dir, "Lina_all.csv")
                    tfidf_vectorizer_april_path = os.path.join(dir, "tfidf_vectorizer_april.pickle")
                    tfidf_matrix_train_april_path = os.path.join(dir, "tfidf_matrix_train_april.pickle")
                    response, line_id = talk_to_lina(var, Lina_all_path, tfidf_vectorizer_april_path,
                                                     tfidf_matrix_train_april_path)

                elif option == 1:
                    response, line_id = talk_to_lina(var, get_relative_path("action_conversation.csv"),
                                                     get_relative_path('tfidf_vectorizer_action.pickle'),
                                                     get_relative_path('tfidf_matrix_train_action.pickle'))

                elif option == 2:
                    response, line_id = talk_to_lina(var, get_relative_path("animation_conversation.csv"),
                                                     get_relative_path('tfidf_vectorizer_animation.pickle'),
                                                     get_relative_path('tfidf_matrix_train_animation.pickle'))

                elif option == 3:
                    response, line_id = talk_to_lina(var, get_relative_path("comedy_conversation.csv"),
                                                     get_relative_path('tfidf_vectorizer_comedy.pickle'),
                                                     get_relative_path('tfidf_matrix_train_comedy.pickle'))

                elif option == 4:
                    response, line_id = talk_to_lina(var, get_relative_path("crime_conversation.csv"),
                                                     get_relative_path('tfidf_vectorizer_crime.pickle'),
                                                     get_relative_path('tfidf_matrix_train_crime.pickle'))

                elif option == 5:
                    response, line_id = talk_to_lina(var, get_relative_path("drama_conversation.csv"),
                                                     get_relative_path('tfidf_vectorizer_drama.pickle'),
                                                     get_relative_path('tfidf_matrix_train_drama.pickle'))

                elif option == 6:
                    response, line_id = talk_to_lina(var, get_relative_path("fantasy_conversation.csv"),
                                                     get_relative_path('tfidf_vectorizer_fantasy.pickle'),
                                                     get_relative_path('tfidf_matrix_train_fantasy.pickle'))

                elif option == 7:
                    response, line_id = talk_to_lina(var, get_relative_path("film-noir.csv_conversation.csv"),
                                                     get_relative_path('tfidf_vectorizer_film-noir.pickle'),
                                                     get_relative_path('tfidf_matrix_train_film-noir.pickle'))

                elif option == 8:
                    response, line_id = talk_to_lina(var, get_relative_path("horror_conversation.csv"),
                                                     get_relative_path('tfidf_vectorizer_horror.pickle'),
                                                     get_relative_path('tfidf_matrix_train_horror.pickle'))

                elif option == 9:
                    response, line_id = talk_to_lina(var, get_relative_path("romance_conversation.csv"),
                                                     get_relative_path('tfidf_vectorizer_romance.pickle'),
                                                     get_relative_path('tfidf_matrix_train_romance.pickle'))

                elif option == 10:
                    response, line_id = talk_to_lina(var, get_relative_path("sci-fi_conversation.csv"),
                                                     get_relative_path('tfidf_vectorizer_sci-fi.pickle'),
                                                     get_relative_path('tfidf_matrix_train_sci-fi.pickle'))

                elif option == 11:
                    response, line_id = talk_to_lina(var, get_relative_path("war_conversation.csv"),
                                                     get_relative_path('tfidf_vectorizer_war.pickle'),
                                                     get_relative_path('tfidf_matrix_train_war.pickle'))

                print

                print ("Lina :  " + response)
                edit_option = raw_input("Do you need to edit the response of the question ?? y/n :")
            
                if(edit_option=="y") :
                      new_sentence = raw_input("Your edit to the previous Lina's response : ")
                      edit_real_time(new_sentence , option , line_id)
                print
            return "message", response.capitalize().strip().split('.')[0] + '.'

        else:
            print ("youssef intent", intentType)
            return   intentType #, action
    else:
        # print ("intent", result)
        print ("anwar intent", result)
        return result      #anwar intent


def get_relative_path(filename):
    conversations_dir = os.path.join(dir, "Conversations")
    relative_path = os.path.join(conversations_dir, filename)
    return relative_path


