#Visualization bar chart
from collections import Counter
import pandas as pd
import matplotlib.colors as mcolors
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from gensim.models.nmf import Nmf
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import preprocess

def word_cloud(topics, n_topics):
    
    #Additional stopwords and colors
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 
                   'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 
                   'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 
                   'may', 'take', 'come'])

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  
    
    #Cloud object
    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=2000,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)
        
    #For 1 topic    
    if n_topics == 1:
        topic_words = dict(topics[0][1])
        i = 0
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(1), fontdict=dict(size=16))
        plt.gca().axis('off')
        
        return plt
        
    #For topics from 2-8
    else:   
        #Set figure dimension
        if n_topics == 2:
            columns = 1
            rows = 2
        elif n_topics == 3:
            columns = 1
            rows = 3
        elif n_topics == 4:
            columns = 2
            rows = 2
        elif n_topics == 5:
            columns = 2
            rows = 3
        elif n_topics == 6:
            columns = 2
            rows = 3
        else:
            columns = 2
            rows = 4
        
        #Visualization  
        fig, axes = plt.subplots(columns,rows, figsize=(10,10), sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            #Make sure it stops trying to fill up the grid before it runs out of topics
            if i+1 > n_topics :
                break
            else:
                fig.add_subplot(ax)
                topic_words = dict(topics[i][1])
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=16))
                plt.gca().axis('off')

        fig.suptitle('Word Cloud for each topic', fontsize=22, y=1.05)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()
        
        return plt


def bar_chart(topics, data, n_topics):
    
    #Create dataframe with info
    data_flat = [w for w_list in data for w in w_list]
    counter = Counter(data_flat)
    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])
    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

    
    #Colors
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    #For 1 topic    
    if n_topics == 1:
        fig, ax = plt.subplots(figsize=(5,5), sharey=True, dpi=160)
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==0,:], color=cols[0], width=0.5, label='Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==0, :], color=cols[1], width=0.1, label='Importance')
        ax.set_ylabel('Word Count', color=cols[0])
        ax_twin.set_ylim(0, 0.06); ax.set_ylim(0, 50)
        ax.set_title('Topic: ' + str(1), color=cols[0], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left', fontsize=10); ax_twin.legend(loc='upper right', fontsize=10)

        return plt

    else:   
        #Set figure dimension
        if n_topics == 2:
            columns = 1
            rows = 2
        elif n_topics == 3:
            columns = 1
            rows = 3
        elif n_topics == 4:
            columns = 2
            rows = 2
        elif n_topics == 5:
            columns = 2
            rows = 3
        elif n_topics == 6:
            columns = 2
            rows = 3
        else:
            columns = 3
            rows = 3

        #Visualization  
        fig, axes = plt.subplots(columns, rows, figsize=(15,10), sharey=True, dpi=160)

        for i, ax in enumerate(axes.flatten()):
            if i+1 > n_topics:
                break
            else:
                ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i,:], color=cols[i+2], width=0.5, label='Count')
                ax_twin = ax.twinx()
                ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.1, label='Importance')
                ax.set_ylabel('Word Count', color=cols[0])
                ax.set_yticklabels(range(0,60,10),  fontsize=17)
                ax_twin.set_ylim(0, 0.06); ax.set_ylim(0, 50)
                ax.set_title('Topic: ' + str(i+1), color=cols[0], fontsize=16)
                ax.yaxis.label.set_size(17)
                ax.tick_params(axis='y', left=False)
                ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= "right", fontsize=17)
                ax.legend(loc='upper left', fontsize=8); ax_twin.legend(loc='upper right', fontsize=8)

        fig.tight_layout(w_pad=2)    
        fig.suptitle('Word Count and Importance for each topic', fontsize=22, y=1.05)    
        plt.show()

#data for best number of topics
def n_topics_data(doc, mode=None):
    if mode == "Lemmatization":
        return preprocess.bigrams(preprocess.preprocess_text(doc,norm = "Lemmatization"))
    elif mode == "Stemming":
        return preprocess.bigrams(preprocess.preprocess_text(doc,norm = "Stemming"))
    else:
        return preprocess.bigrams(preprocess.preprocess_text(doc))

#Plot to detect best number of topics
def best_n_topics(corpus, dictionary, doc, model="Latent Dirichlet Allocation", data_mode=None):
    data = doc
    tweets_coherence = []
    
    #loop for performance metric
    if model=="Latent Dirichlet Allocation":
        for nb_topics in range(1,9):
            lda = LdaModel(corpus, num_topics = nb_topics, id2word = dictionary, passes=1, random_state=42)
            cohm = CoherenceModel(model=lda, texts=data, corpus=corpus, dictionary=dictionary, coherence='c_v')
            coh = cohm.get_coherence()
            tweets_coherence.append(coh)     
    else:
        for nb_topics in range(1,9):
            nmf_model = Nmf(corpus, num_topics = nb_topics, id2word = dictionary, passes=1, random_state=42)
            cohm = CoherenceModel(model=nmf_model, texts=data, corpus=corpus, dictionary=dictionary, coherence='c_v')
            coh = cohm.get_coherence()
            tweets_coherence.append(coh)

    #Best_n_topics
    number_topics = tweets_coherence.index(max(tweets_coherence))+1
    
    # visualize coherence
    plt.figure(figsize=(5,5))
    plt.title("Coherence for each nº of topic", fontdict = {'fontsize' : 8})
    plt.plot(range(1,9), tweets_coherence)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=10)
    
    return plt 













