#Models
from gensim.models.ldamodel import LdaModel
from gensim.models.nmf import Nmf
from gensim.models.hdpmodel import HdpModel

#Models
def topic_models(corpus, dictionary, n_topics=0, kind="Latent Dirichlet Allocation"):
    if kind=="Latent Dirichlet Allocation":
        model = LdaModel(corpus=corpus, id2word=dictionary,num_topics=n_topics, 
                    alpha="auto", eta='auto', iterations=50, chunksize=100, passes=50, random_state=123)
        
        topics= model.show_topics(formatted=False, num_words=10)
    
    elif kind=="Non-Negative Matrix Factorization":
        model = Nmf(corpus, id2word=dictionary, num_topics=n_topics,
                     passes=50, w_max_iter=50, random_state=123)
        
        topics= model.show_topics(formatted=False, num_words=10)
    
    else:
        model = HdpModel(corpus, id2word=dictionary,
                         chunksize=100,random_state=123)
        
        topics= model.show_topics(formatted=False, num_words=10, num_topics=n_topics)
        
    return topics