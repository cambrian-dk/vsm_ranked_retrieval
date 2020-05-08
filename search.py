#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import pickle
import os
from math import floor, sqrt, log
from collections import defaultdict
import heapq

#constants
STEMMER = nltk.stem.porter.PorterStemmer()
MAX_RESULTS = 10

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

#file processing

def readQueries(queries_file):
    """Parses the query file into a list of separated queries
    
    Arguments:
        queries_file {string} -- the path to the queries file
    
    Returns:
        list -- the list of split queries
    """
    queries = open(queries_file, 'r').read().splitlines()
    return queries


def getLengthsDict(postingsReader):
    """gets the lengthsDict from the postings file (it is the first item in the postings file)
    
    Arguments:
        postingsReader {file object} -- the file object obtained by opening the postings file
    
    Returns:
        dict -- the length of each document, represented by a dictionary
    """
    lengths = pickle.load(postingsReader)
    return lengths

#query parsing

#tokenizes the free text query
def tokenizeQuery(query):
    tokens = query.split() 
    #casefold, then stem the tokens
    return list(map(lambda y:STEMMER.stem(y), list(map(lambda x:x.lower(), tokens))))

def processQuery(query, dictionary, postingsReader, lengths):
    """Processes a single query given the dictionary, postings and length
    
    Arguments:
        query {string} -- the free text query to be processed
        dictionary {dict} -- the dictionary of tokens
        postingsReader {file object} -- the file object obtained by opening the postings file
        lengths {dict} -- the length of each document, represented by a dictionary
    
    Returns:
        list -- the list of most relevant results (a maximum of MAX_RESULTS)
    """

    tokens = tokenizeQuery(query)
    scores = defaultdict(float)
    for token in tokens:
        #calculate weight of token
        qtf = 1 + log(tokens.count(token), 10)
        if token in dictionary:
            idf = dictionary[token][1]
        else:
            #the token does not exist in the dictionary, so the idf is 0 and we can skip this token
            idf = 0
            print("skip")
            print(token)
            continue
        qidftf = qtf * idf
        postingsReader.seek(dictionary[token][2])
        postings = pickle.load(postingsReader)
        for docId, weightedTermFreq in postings.items():
            scores[docId] += qidftf * weightedTermFreq
    
    for docId, total in scores.items():
        scores[docId] /= lengths[docId]
    
    #the documents are sorted by score, and documentID is used to tiebreak
    return heapq.nlargest(MAX_RESULTS, scores, key = lambda docId : (scores[docId], -docId))

    

def runSearch(dict_file, postings_file, queries_file, results_file):  
    """ using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    
    Arguments:
        dict_file {string} -- path to the dictionary
        postings_file {string} -- path to the postings file
        queries_file {string} -- path to the queries file
        results_file {string} -- path to the results file
    """
    print('running search on the queries...')
    # This is an empty method
    # Pls implement your code in below

    #pre-processing
    queries = readQueries(queries_file)
    dictionary = pickle.load(open(dict_file, 'rb'))
    postingsReader = open(postings_file, 'rb')
    lengths = getLengthsDict(postingsReader)

    #clear any results files
    if os.path.isfile(results_file):
        #clear any previous text
        results = open(results_file, "r+")
        results.truncate(0)
        results.close()
    results = open(results_file, "a")
    #only write newline characters between consecutive lines
    newline = ''
    #iterate through the queries and process each of them
    for query in queries:
        answer = processQuery(query, dictionary, postingsReader, lengths)
        results.write(newline)
        results.write(" ".join(map(str, answer)))
        newline = '\n'
        
    results.close()



dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

runSearch(dictionary_file, postings_file, file_of_queries, file_of_output)
