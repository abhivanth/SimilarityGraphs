import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize(df, text_column, id_column, name):
    tfidf_vectorizer = TfidfVectorizer(max_features=2000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
    tfidf_matrix = tfidf_matrix.toarray()
    s2orc_tfidf = pd.DataFrame(tfidf_matrix, columns=tfidf_vectorizer.get_feature_names_out(),
                               index=df[text_column].index)
    s2orc_tfidf['id'] = df[id_column]
    print(s2orc_tfidf.shape)
    s2orc_tfidf.to_csv(f'text_embeddings/{name}_embeddings_tfidf.csv', index=False)

