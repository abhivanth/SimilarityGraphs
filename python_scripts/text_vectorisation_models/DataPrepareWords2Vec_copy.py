import pandas as pd
from gensim.models import Word2Vec
from python_scripts.text_vectorisation_models.PrepareAmazonDataSet import load_amazon_data
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import nltk

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens


combined_text = load_amazon_data()

combined_text['title_x'] = combined_text['title_x'].apply(preprocess_text)
combined_text['text'] = combined_text['text'].apply(preprocess_text)

sentences = combined_text['title_x'].tolist() + combined_text['text'].tolist()

word2vec_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)


def get_document_vector(doc, model):
    vectors = [model.wv[word] for word in doc if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


combined_text['title_vector'] = combined_text['title_x'].apply(lambda x: get_document_vector(x, word2vec_model))
combined_text['text_vector'] = combined_text['text'].apply(lambda x: get_document_vector(x, word2vec_model))

combined_text['document_vector'] = combined_text.apply(
    lambda row: np.concatenate([row['title_vector'], row['text_vector']]), axis=1)

vector_dim = len(combined_text['document_vector'][0])
vector_columns = [f'vector_{i}' for i in range(vector_dim)]
document_vectors_df = pd.DataFrame(combined_text['document_vector'].tolist(), columns=vector_columns)

result_df = pd.concat(
    [combined_text.drop(columns=['title_vector', 'text_vector', 'document_vector']), document_vectors_df], axis=1)

result_df.to_csv("amazon_word2vec.csv", index=False)
