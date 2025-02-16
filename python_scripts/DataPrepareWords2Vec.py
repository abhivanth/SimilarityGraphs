import pandas as pd
from gensim.models import Word2Vec
from python_scripts.PrepareAmazonDataSet import load_amazon_data
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens


def get_document_vector(doc, model):
    doc_title = doc['title_x'].tolist()
    doc_text = doc['text'].tolist()
    word_vectors_title = []
    word_vectors_text = []
    for words in doc_title:
        inner_list = []
        for word in words:
            if word in model.wv:
                inner_list.append(model.wv[word])
            else:
                inner_list.append(0)
        word_vectors_title.append(inner_list)

    for words in doc_text:
        inner_list = []
        for word in words:
            if word in model.wv:
                inner_list.append(model.wv[word])
            else:
                inner_list.append(0)
        word_vectors_text.append(inner_list)

    word_vectors_title_df = pd.DataFrame(word_vectors_title)
    word_vectors_text_df = pd.DataFrame(word_vectors_text)
    combined_vector = pd.concat([word_vectors_title_df.fillna(0), word_vectors_text_df.fillna(0)], axis=1)
    combined_vector = np.array(combined_vector)
    return np.mean(combined_vector, axis=0)


combined_text = load_amazon_data()
combined_text_title = combined_text['title_x'].apply(preprocess_text)
combined_text_text = combined_text['text'].apply(preprocess_text)

result_df = pd.concat([combined_text_title, combined_text_text], axis=1)
word2vec_model = Word2Vec(sentences=result_df, vector_size=100, window=5, min_count=1, workers=4)

result_df_vector = get_document_vector(result_df, word2vec_model)
np.savetxt("amazon_word2vec.csv", result_df_vector, delimiter=",")
