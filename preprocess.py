#Preprocess
#Libraries dependencies
import re
from nltk.stem import (WordNetLemmatizer, PorterStemmer)
import nltk
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary

#Common steps
def preprocess_text(document, norm=None):
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        nltk.download('stopwords')
        en_stop = set(nltk.corpus.stopwords.words('english'))
    
        document = re.sub(r'\W', ' ', str(document))
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = document.lower()
        
        tokens = document.split()
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word)  > 3]
        
        if norm == "Lemmatization":
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        elif norm == "Stemming":
            tokens = [stemmer.stem(word) for word in tokens]
       
        return [tokens]

#Bigrams
def bigrams(data):
    bigram = gensim.models.Phrases(data, min_count=3, threshold=3)
    data_bigram = [x for x in bigram[data]]
    
    return data_bigram

#Feature engineering
#Dictionary
def dic(data):
    dictionary = corpora.Dictionary(data)
    return dictionary
    
#Corpus
def corpus(data, extraction="TF-IDF"):
    ##Using term frequency
    dictionary = dic(data)
    BoW_corpus = [dictionary.doc2bow(doc) for doc in data]
    
    #Using tfidf
    BoW_corpus_tfidf = gensim.models.TfidfModel(BoW_corpus, smartirs='ntc')
    tfidf_corpus = BoW_corpus_tfidf[BoW_corpus]
    
    if extraction=="TF-IDF":
        return  tfidf_corpus
    
    elif extraction=="Count":
        return BoW_corpus