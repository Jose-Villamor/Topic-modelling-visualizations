import streamlit as st
import preprocess
import models
import visualization

st.set_option('deprecation.showPyplotGlobalUse', False)

def options():
    config = {}
    
    normalization = st.radio("Text normalization technique", ("None",  "Stemming", "Lemmatization"))
    config["normalization"] = normalization

    extraction = st.radio("BOW feature extraction method", ("Count", "TF-IDF"))
    config["extraction"] = extraction

    model = st.radio("Choose model", ("Latent Dirichlet Allocation", "Non-Negative Matrix Factorization", "Hierarchical Dirichlet Process"))
    config["model"] = model

    return config

def n_topics_options():
    number_topics = {}

    n_topics = st.slider("Number of topics", min_value=1, max_value=8, value=4, step=1)
    number_topics["n_topics"] = n_topics

    return number_topics

st.set_page_config(
    page_title="Text Summarization",
    layout="centered",
    initial_sidebar_state="expanded")

st.subheader("Jose Villamor")

html_temp = """
    <div style="background:#fab300 ;padding:10px">
    <h2 style="color:black;text-align:center;"> Topic modelling visualization </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html = True)
     
#st.image("summaries.png", use_column_width=True)  
     
st.write("Topic modelling is an unsupervised machine learning technique where we cluster words into different themes. This web app performs topic modelling using various preprocessing, feature engineering and modelling approaches.") 
st.write("Insert a text select the parameters and click the button(last one) to visualize the topics. There is the option to predict the best number of topics by clicking the first button.")

st.write("**If you want to know more about the project or others that i have done visit my github account: https://jose-villamor.github.io/Portfolio_website/portfolio.html**")

st.write("**Insert text**")
document = st.text_area("5000 Max Characters ", height=400, max_chars=5000)

st.write("**Select model parameters**")
parameters = options()


st.write("**If you want to know the best number of topics click the button below (Not available for Hierarchical Dirichlet Process model). Can take several minutes to compute.**")
st.write("Select the one with higher score")
if st.button("Best number of topics"):

    data = visualization.n_topics_data(document, parameters["normalization"])   

    prepro_doc = preprocess.preprocess_text(document, norm=parameters["normalization"])
    prepro_doc_bigr = preprocess.bigrams(prepro_doc)

    dictionary = preprocess.dic(prepro_doc_bigr)
    corpus = preprocess.corpus(prepro_doc_bigr, extraction=parameters["extraction"])

    best_topics = visualization.best_n_topics(corpus, dictionary, doc=data, model=parameters["model"])

    st.pyplot(best_topics)

st.write("**Select number of topics**")
n_of_topics = n_topics_options()

#Buttons
st.write("**Click the button to visualize the topics**")
if st.button("Visualize topics"):
    prepro_doc = preprocess.preprocess_text(document, norm=parameters["normalization"])
    prepro_doc_bigr = preprocess.bigrams(prepro_doc)

    dictionary = preprocess.dic(prepro_doc_bigr)
    corpus = preprocess.corpus(prepro_doc_bigr, extraction=parameters["extraction"])

    show_topics = models.topic_models(corpus, dictionary, n_topics=n_of_topics["n_topics"], kind=parameters["model"])

    plot_cloud = visualization.word_cloud(topics=show_topics, n_topics=n_of_topics["n_topics"])
    st.pyplot(plot_cloud)

    plot_bar = visualization.bar_chart(topics=show_topics, data=prepro_doc_bigr, n_topics=n_of_topics["n_topics"])
    st.pyplot(plot_bar)

   
