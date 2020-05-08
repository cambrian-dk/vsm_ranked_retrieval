## Vector Space Model for Ranked Retrieval

Python programs to index and process queries using the Vector Space Model (VSM), developed to test on the Reuters corpus from the Natural Language Toolkit.


### Prerequisites

Written in Python 3.8.2

[Install nltk for python](https://www.nltk.org/install.html), then:

Import the Reuters Corpus:
```
from nltk.corpus import reuters
```



## Format

Queries: one text query per line e.g. 

```
world news
reserve stockpile
global warming and pollution
stock market crash
virus epidemic
```

Results: ranked list of documents from highest to lowest for each query e.g.
```
8402 1989 10815 4761 8429 9668 1719 4959 10685 3377
12433 12817 11801 283 6983 4062 7188 2098 13185 11100
9570 14340 12935 241 8213 1094 10860 8554 12009 11392
11384 129 3507 8533 368 2176 10191 10506 10471 9015
798 2226 7940
```

## Running

Indexing: 
```
python index.py -i dataset-file -d dictionary-file -p postings-file
```

Searching: 
```
python search.py -d dictionary-file -p postings-file -q query-file -o output-file-of-results
```

Sample files are included in the repo...


## The program

### Indexing:

We iterate through the documents and calculate the values to be put in the dictionary and postings files.
A separate class myDict contains:
-the main dictionary
-a reverse idDictionary mapping ids to their respective tokens
    -this is because tokens are referred to by their id for most of the calculations
-the filename to write the dictionary to when we save the dictionary

In the main dictionary, an entry is structured as a dictionary entry, with the keys as the tokens,
and the values as tuples of: (the token id, the document frequency, the byte location of
the postings in the pickled postings file).

After iterating through the whole document, we calculated the weighted term frequency and length and replace 
the original values in the document vector. Performing the calculation here saves time from the searching operation.

In the postings, an entry is structured as a dictionary of dictionaries, the first layer of keys is the tokenId 
as obtained from the dictionary, and that maps to a dictionary where the keys are the document ID, and the values are
the weighted token frequency as calculated previously.

After iterating through all documents, we calculate the inverted document frequency and replace it into the dictionary.

### Searching:

Each free-text search query is case-folded and stemmed as in the indexing phase. We follow the [SMART notation](https://nlp.stanford.edu/IR-book/html/htmledition/document-and-query-weighting-schemes-1.html) to calculate the lnc.ltc score. For query tokens that don't exist in the dictionary, we ignore them, as they are unlikely to 
be useful in ranking the resulting documents. After normalising the document scores, we use a max heap to get the top 10 most relevant
documents, sorted by score, and then increasing docId.