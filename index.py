#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import os
import pickle
import linecache
from collections import defaultdict
from math import log, sqrt

#constants
STEMMER = nltk.stem.porter.PorterStemmer()

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

def build_index(in_dir, out_dict, out_postings):
    """This function builds the dictionary and postings files from the directory.
    
    Arguments:
        in_dir {string} -- path to the directory containing the text for indexing
        out_dict {string} -- path to where the dictionary file will be written to
        out_postings {string} -- path to where the postings file will be written to
    """
    print('indexing...')
    # This is an empty method
    # Pls implement your code in below

    # initialize the dictionary & postings container
    dictionary = myDict(out_dict)
    postings = myPostings()

    # length will be a dictionary mapping document IDs to their lengths
    lengths = dict()

    # iterate through each document in sequence
    docList = [int(txtFile) for txtFile in os.listdir(in_dir)]
    docList.sort()

    # parse each document in the list
    for docName in docList:
        parse_document(docName, dictionary, postings, in_dir, lengths)

    postings.savePostingsToFile(lengths, out_postings, dictionary)

    #we can calculate the IDF for each token during indexing
    calculateIDF(dictionary.getDictionary(), len(docList))

    dictionary.save()
    


#parses a document into the dictionary and postings lists.
def parse_document(docId, dictionary, postings, in_dir, lengths):
    """This function parses a single document, adding the resulting vector to the dictionary and postings lists.
    
    Arguments:
        docId {int} -- the ID of the document to be parsed (assuming all document names are integers)
        dictionary {myDict} -- the myDict object containing the main dictionary
        postings {myPostings} -- the myPostings object containing the postings
        in_dir {string} -- path to the directory containing the text for indexing
        lengths {dict} -- the dictionary mapping document IDs to their lengths
    """

    fileName = str(docId)
    docVector = defaultdict(int)
    lineNum = 1
    
    line = linecache.getline(os.path.join(in_dir, fileName), 1)
    while(line != ''):
        #parse the line for tokens and add to the document's vector.
        parseLine(line, docVector)
        lineNum += 1
        line = linecache.getline(os.path.join(in_dir, fileName), lineNum)
    
    #calculate term frequency for the document, replacing term frequency with the weighted tf-idf (before normalization)
    calculateTermFrequency(docVector)

    #obtain the length of the vector and add it to the length array
    lengths[docId] = calculateVectorLength(docVector)

    #record the data in the dictionary and postings
    recordData(docId, dictionary, postings, docVector)

def parseLine(line, docVector):
    """This function parses a line into tokens and adding their frequencies to the document vector.
    
    Arguments:
        line {string} -- the line to be tokenized
        docVector {defaultdict} -- the document vector represented as a dictionary
    """
    sentences = nltk.sent_tokenize(line)
    for sentence in sentences:
        #use the nltk word tokenizer 
        words = nltk.word_tokenize(sentence)

        #lower and stem each word
        for word in words:
            word = word.lower()
            word = STEMMER.stem(word)

            #add it to the document vector (defaultdict handles new entries automatically)
            docVector[word] += 1

def calculateTermFrequency(docVector):
    """Calculates and replaces the term frequency for the document vector with the weighted value
    
    Arguments:
        docVector {[type]} -- [description]
    """
    for key, value in docVector.items():
        if value != 0:
            docVector[key] = 1 + log(value, 10)
        else:
            #protecting against invalid log values
            docVector[key] = 0

def calculateVectorLength(docVector):
    """Calculates and returns the length of a document vector for cosine normalization
    
    Arguments:
        docVector {dictionary} -- the document vector represented as a dictionary
    
    Returns:
        float -- the normalization factor for the vector
    """
    sum = 0
    for key, value in docVector.items():
        sum += value**2
    return sqrt(sum)

def calculateIDF(dictionary, N):
    """Calculates and replaces the document frequency with the weighted inverse document frequency
    
    Arguments:
        dictionary {dict} -- the dictionary containing the document frequencies
        N {int} -- the total document number
    """
    for token, information in dictionary.items():
        information[myDict.DOCUMENT_FREQUENCY] = log(N/information[myDict.DOCUMENT_FREQUENCY], 10)

def recordData(docId, dictionary, postings, docVector):
    """Records the data from a document vector into the dictionary and postings lists
    
    Arguments:
        docId {int} -- the document id
        dictionary {myDict} -- the main dictionary 
        postings {myPostings} -- the main postings list
        docVector {dict} -- a document vector represented by a dictionary
    """
    for token, weightedTermFreq in docVector.items():
        tokenId = dictionary.addToken(token)
        postings.addTuple(docId, tokenId, weightedTermFreq)


#additional data structures
class myDict():
    TOKEN_ID = 0
    DOCUMENT_FREQUENCY = 1
    POSTING_START = 2
    ID_START = 1

    def __init__(self, fileName):
        self.dictionary = dict()
        self.idDictionary = dict()
        self.fileName = fileName
        self.id = self.ID_START
    
    def getIdDictionary(self):
        return self.idDictionary

    def getDictionary(self):
        return self.dictionary

    #updates the location of a token given the tokenId
    def updateLocation(self, tokenId, location):
        token = self.idDictionary[tokenId]
        self.dictionary[token][self.POSTING_START] = location

    #adds a vector entry(token) to the dictionary, returning its ID in the dictionary
    def addToken(self, token):
        """Adds a token to the dictionary, returning its ID in the dictionary
        
        Arguments:
            token {string} -- the token to be added to the dictionary
        
        Returns:
            int -- the ID of the token in the dictionary
        """
        return_id = 0
        if token in self.dictionary:
            #the token exists
            return_id = self.dictionary[token][self.TOKEN_ID] 
            self.dictionary[token][self.DOCUMENT_FREQUENCY] += 1
        else:
            #the token is new so we assign it a new id
            self.dictionary[token] = [self.id, 1, 0]
            self.idDictionary[self.id] = token
            return_id = self.id
            self.id += 1

        return return_id
    
    #saves the dictionary to file
    def save(self):
        """This function saves the dictionary to file
        We use pickling to save the dictionary.
    
        """

        if os.path.isfile(self.fileName):
            #clear any previous text
            out = open(self.fileName, "r+")
            out.truncate(0)
            out.close()

        with open(self.fileName, 'wb') as f:
            pickle.dump(self.dictionary, f)


class myPostings():

    #postings are structure as a double layer dictionary, first tokenId - postings list,
    #then documentName to term frequency/inverse document frequency

    def __init__(self):
        self.postings = dict()
    
    def addTuple(self, docName, tokenId, weightedTokenFreq):
        """Adds the document name and weighted tokenFrequency tuple to the postings list of the respective tokenId
        
        Arguments:
            docName {int} -- the name of the document
            tokenId {int} -- the id of the token to be added (this represents the token)
            weightedTokenFreq {float} -- the calculated weighted token frequency
        """
        if tokenId in self.postings:
            #the tokenId already exists, so we just add the key and value to it
            posting = self.postings[tokenId]
            posting[docName] = weightedTokenFreq
        else:
            #we initialize a new dictionary for the postings list of that tokenId
            self.postings[tokenId] = dict()
            self.postings[tokenId][docName] = weightedTokenFreq

    def savePostingsToFile(self, lengths, out_postings, dictionary):
        """Saves the postings list to file. We use pickling to store the postings list.
        We store the lengths along with the postings as well.
        
        Arguments:
            lengths {dict} -- the length of each document represented by a dictionary
            out_postings {string} -- the path to write the posting to
            dictionary {myDict} -- we need the dictionary to update the stored location of the postings
        """
        if os.path.isfile(out_postings):
            #clear any previous text
            out = open(out_postings, "r+")
            out.truncate(0)
            out.close()
        out = open(out_postings, "ab")
        pickle.dump(lengths, out)
        for tokenId in dictionary.getIdDictionary():
            posting = self.postings[tokenId]
            dictionary.updateLocation(tokenId, out.tell())
            pickle.dump(posting, out)
        out.close()
    



#main method


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        if not a.endswith('/'):
            #bulletproofing 
            a = a + '/'
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)