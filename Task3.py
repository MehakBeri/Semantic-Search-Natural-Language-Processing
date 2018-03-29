import os
import json
import nltk
import Constants
import SolrCommunicator

from nltk.wsd import lesk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Constants import Constants
from SolrCommunicator import SolrCommunicator
from nltk.parse.stanford import StanfordDependencyParser

#Read All FIles from directory
#Read Every Sentence
#Find POS for every Word in Every Sentence
#Depending on POS and Word Find its Synonym, Hyepernum, lemma

class Task3:

    path_to_jar = 'C://Users//Mehak Beri//Desktop//MS//fall 17//nlp//Final Project//stanford-corenlp-full-2017-06-09//stanford-corenlp-3.8.0.jar'
    path_to_models_jar = 'C://Users//Mehak Beri//Desktop//MS//fall 17//nlp//Final Project//stanford-corenlp-full-2017-06-09//stanford-corenlp-3.8.0-models.jar'
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

    def __init__(self):
        Task3.Constant = Constants()
        Task3.SolrCommunicator = SolrCommunicator()

    #ReadCorpus(Constant.location, SolrCommunicator)
    def getHeadWord(self, line):
        result = self.dependency_parser.raw_parse(line)
        dep = result.__next__()
        if dep is not None and dep.triples() is not None and len(list(dep.triples())) > 0 :
            #print("headWord")
            return list(dep.triples())[0][0][0]
        else:
            return ""

    def getHolonyms(self, word):
        holonyms = []
        for ss in wordnet.synsets(word):
            for holo in ss.part_holonyms():
                holonyms.append(holo.name().split('.')[0]);
        return holonyms

    def getMeronyms(self, word):
        meronyms = []
        for ss in wordnet.synsets(word):
            for mero in ss.part_meronyms():
                meronyms.append(mero.name().split('.')[0]);
        return meronyms

    def getHypernyms(self, word):
        hypernyms = []
        for ss in wordnet.synsets(word):
            for hyper in ss.hypernyms():
                hypernyms.append(hyper.name().split('.')[0]);
        return hypernyms

    def getHyponyms(self, word):
        hyponyms = []
        for ss in wordnet.synsets(word):
            for hypo in ss.hyponyms():
                hyponyms.append(hypo.name().split('.')[0]);
        return hyponyms

    def returnSolrDataFormat(self, filename, lineNo, title, line, words, posTags,
                             lemmas, lesks, stemma, hypernyms, hyponyms, meronyms, holonyms, headWord):
        return {
            'id' : 'Doc_'+filename+'_sentence_'+str(lineNo),
            'title' : title,
            'sentence' : line,
            'words' : words,
            'posTags' : posTags,
            'lemmas' : lemmas,
            'lesks':lesks,
            'stemma' : stemma,
            'hypernyms' : hypernyms,
            'hyponyms' : hyponyms,
            'meronyms' : meronyms,
            'holonyms' : holonyms,
            'headWord' : headWord
        };

    def addDataToSolr(self, data):
        if len(data) >0 :    
            if Task3.SolrCommunicator.addDataToSolr(data) is None:
                print('Error Caused While Adding Data to solr')
            else:
                print('Data Successfully Added to Solr')

    def ReadCorpus(self):
        lineNo = 0
        title = ""
        index = 0
        finalIndex = 0
        data = []
        posTags = []
        wordPosTag = []
        lemmatizer = WordNetLemmatizer()
        ps = PorterStemmer()
        for filename in os.listdir(Task3.Constant.location):
            if index >= Task3.Constant.solrBatchSize :
                self.addDataToSolr(data)
                data = []
                finalIndex = finalIndex + index
                index = 0

            with open(Task3.Constant.location + filename) as file:
                read_data = file.readlines()
                lineNo = 0
                filedata = ""
                for i in range(1,len(read_data)):
                    filedata+=read_data[i].replace("\n", "")

            for line in sent_tokenize(filedata):
                #find pos of every word in the sentence using nltk
                words = word_tokenize(line.replace('\n', ''))
                if words is not None and len(words) > 0 and len(words)<50:
                    if lineNo == 0 :
                        title = line
                    lineNo += 1
                    wordPosTag = []
                    lemmas = []
                    lesks=[]
                    stemma = []
                    hypernyms = []
                    hyponyms = []
                    meronyms = []
                    holonyms = []
                    
                    posTags = nltk.pos_tag(words)
                    headWord = self.getHeadWord(line)
                    for posTagIndex in range(0,len(posTags)):
                        wordPosTag.append(posTags[posTagIndex][0] + " " + posTags[posTagIndex][1])
                        #wordPosTag.append(posTags[posTagIndex][1])
                        lemmas.append(lemmatizer.lemmatize(words[posTagIndex]))
                        lesks.append(str(lesk(words, words[posTagIndex])))
                        stemma.append(ps.stem(words[posTagIndex]))
                        hypernyms.append(self.getHypernyms(words[posTagIndex]))
                        hyponyms.append(self.getHyponyms(words[posTagIndex]))
                        meronyms.append(self.getMeronyms(words[posTagIndex]))
                        holonyms.append(self.getHolonyms(words[posTagIndex]))
                    data.append(
                        self.returnSolrDataFormat(
                            filename, lineNo, title, line, words, 
                            wordPosTag, lemmas, lesks, stemma, 
                            hypernyms, hyponyms, meronyms, holonyms, headWord
                        )
                    );

            index = index + 1
        self.addDataToSolr(data)
        print(finalIndex)
        return data

    def analyzeSingleSentenceTask4(self, line):
        data = []
        posTags = []
        wordPosTag = []
        lemmas = []
        lesks=[]
        stemma = []
        hypernyms = []
        hyponyms = []
        meronyms = []
        holonyms = []
        wordsWithoutStopWords = []
        lemmatizer = WordNetLemmatizer()
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        for line in sent_tokenize(line):
            words = word_tokenize(line.replace('\n', ''))
            if words is not None and len(words) > 0:
                posTags = []
                wordPosTag = []
                lemmas = []
                lesks=[]
                stemma = []
                hypernyms = []
                hyponyms = []
                meronyms = []
                holonyms = []
                posTags = nltk.pos_tag(words)
                headWord = self.getHeadWord(line)
                print("Head Word " + str(headWord))
                for posTagIndex in range(0,len(posTags)):
                    if words[posTagIndex] not in stop_words:
                        wordsWithoutStopWords.append(words[posTagIndex])
                        wordPosTag.append(posTags[posTagIndex][0] + " " + posTags[posTagIndex][1])
                        lemmas.append(lemmatizer.lemmatize(words[posTagIndex]))
                        lesks.append(str(lesk(words, words[posTagIndex])))
                        stemma.append(ps.stem(words[posTagIndex]))
                        hypernyms.append(self.getHypernyms(words[posTagIndex]))
                        hyponyms.append(self.getHyponyms(words[posTagIndex]))
                        meronyms.append(self.getMeronyms(words[posTagIndex]))
                        holonyms.append(self.getHolonyms(words[posTagIndex]))
        data=""
        for word in wordsWithoutStopWords:
            data+="(words:"+ word + ")^" + str(Task3.Constant.wordWt) + " "
        for word in wordPosTag:
            data+="(posTags:"+ word + ")^" + str(Task3.Constant.posWt) + " "
        for word in lemmas:
            data+="(lemmas:"+ word + ")^" + str(Task3.Constant.lemmasWt) + " "
        for word in lesks:
            data+="(lesks:"+ (word) + ")^" + str(Task3.Constant.leskWt) + " "
        for word in stemma:
            data+="(stemma:"+ word + ")^" + str(Task3.Constant.stemmaWt) + " "
        for sublist in hypernyms:
            for word in sublist:
                if word is not None:
                    data+="(hypernyms:"+ word + ")^" + str(Task3.Constant.hypernymsWt) + " "
        for sublist in hyponyms:
            for word in sublist:
                if word is not None:
                    data+="(hyponyms:"+ word + ")^" + str(Task3.Constant.hyponymsWt) + " "
        for sublist in meronyms:
            for word in sublist:
                if word is not None:
                    data+="(meronyms:"+ word + ")^" + str(Task3.Constant.meronymsWt) + " "
        for sublist in holonyms:
            for word in sublist:
                if word is not None:
                    data+="(holonyms:"+ word + ")^" + str(Task3.Constant.holonymsWt) + " "
        for sublist in holonyms:
            for word in sublist:
                if headWord is not None:
                    data+="(headWord:"+ headWord + ")^" + str(Task3.Constant.headWordWt) + " "
        print("task 4")
        print("query ")
        print(data)
        return self.searchSolr(data)

    def analyzeSingleSentenceTask3(self, line):
        data = []
        posTags = []
        wordPosTag = []
        lemmas = []
        lesks=[]
        stemma = []
        hypernyms = []
        hyponyms = []
        meronyms = []
        holonyms = []
        lemmatizer = WordNetLemmatizer()
        ps = PorterStemmer()
        for line in sent_tokenize(line):
            words = word_tokenize(line.replace('\n', ''))
            if words is not None and len(words) > 0:
                posTags = []
                wordPosTag = []
                lemmas = []
                lesks=[]
                stemma = []
                hypernyms = []
                hyponyms = []
                meronyms = []
                holonyms = []
                posTags = nltk.pos_tag(words)
                headWord = self.getHeadWord(line)
                for posTagIndex in range(0,len(posTags)):
                    wordPosTag.append(posTags[posTagIndex][1])
                    lemmas.append(lemmatizer.lemmatize(words[posTagIndex]))
                    lesks.append(str(lesk(words,words[posTagIndex])))
                    stemma.append(ps.stem(words[posTagIndex]))
                    hypernyms.append(self.getHypernyms(words[posTagIndex]))
                    hyponyms.append(self.getHyponyms(words[posTagIndex]))
                    meronyms.append(self.getMeronyms(words[posTagIndex]))
                    holonyms.append(self.getHolonyms(words[posTagIndex]))
                data.append(
                    self.returnSolrDataFormat(
                        "filename", "lineNo", "title", line, words, 
                        wordPosTag, lemmas, lesks, stemma, 
                        hypernyms, hyponyms, meronyms, holonyms, headWord
                    )
                );
        data=""
        for word in words:
            data+="words:"+word + " "
        for word in wordPosTag:
            data+="posTags:"+word + " "
        for word in lemmas:
            data+="lemmas:"+word + " "
        for word in lesks:
            data+="lesks:"+(word) + " "
        for word in stemma:
            data+="stemma:"+word + " "
        for sublist in hypernyms:
            for word in sublist:
                if word is not None:
                    data+="hypernyms:"+word + " "
        for sublist in hypernyms:
            for word in sublist:
                if word is not None:
                    data+="hypernyms:"+word + " "
        for sublist in hyponyms:
            for word in sublist:
                if word is not None:
                    data+="hyponyms:"+word + " "
        for sublist in meronyms:
            for word in sublist:
                if word is not None:
                    data+="meronyms:"+word + " "
        for sublist in holonyms:
            for word in sublist:
                if word is not None:
                    data+="holonyms:"+word + " "
        if headWord is not None:
            for word in sublist:
                if word is not None:
                    data+="headWord:"+headWord + " "
        return self.searchSolr(data)

    def analyzeSingleSentenceTask2(self, line):
       
        data = ""
        print('Query: '+ line)
        for line in sent_tokenize(line):
            words = word_tokenize(line.replace('\n', ''))
        for word in words:
            data+="words:"+word + " "
        print(data)
        return self.searchSolr(data)

    def searchSolr(self, data):
#        data=self.escapeSolrArg(data)
        results = Task3.SolrCommunicator.solrSearch(
            data, 'sentence score', '10','score desc'
        )
        if results is not None:
            print("Saw {0} result(s).".format(len(results)))
            for result in results:
                #print("The title is '{0}'.".format(result))
                print("Score : "  + str(result['score']) + "\nSentence : " + result['sentence'][0] + "\n")

        else:
            print('Some error in results')
            


    def escapedSeq(self,term):
#    """ Yield the next string based on the
#        next character (either this char
#        or escaped version """
    # These rules all independent, order of
# escaping doesn't matter
        escapeRules = {
#                '+': r'\+',
#               '-': r'\-',
#               '&': r'\&',
#               '|': r'\|',
#               '!': r'\!',
#               '(': r'\(',
#               ')': r'\)',
#               '{': r'\{',
#               '}': r'\}',
#               '[': r'\[',
#               ']': r'\]',
#               '^': r'\^',
#               '~': r'\~',
#               '*': r'\*',
#               '?': r'\?',
#               ':': r'\:',
#               '"': r'\"',
#               ';': r'\;',
               ' ': r'\ '}
        for char in term:
            if char in escapeRules.keys():
                yield escapeRules[char]
            else:
                yield char

    def escapeSolrArg(self,term):
#    """ Apply escaping to the passed in query terms
#        escaping special characters like : , etc"""
        print(term)
        term = term.replace('\\', r'\\')   # escape \ first
        print("".join([nextStr for nextStr in self.escapedSeq(term)]))
        return "".join([nextStr for nextStr in self.escapedSeq(term)])