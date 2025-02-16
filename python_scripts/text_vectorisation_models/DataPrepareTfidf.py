import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from python_scripts.text_vectorisation_models.PrepareAmazonDataSet import load_amazon_data

combined_text = load_amazon_data()

tfidf_vectorizer = TfidfVectorizer(max_features=2000)  # Limit to 2000 features for richer representation
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=combined_text.index,
                        columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df['rating'] = combined_text['rating']
tfidf_df.to_csv('tfidf_features_with_rating.csv', index=False)

print(combined_text[['text', 'title_x', 'main_category']].head())
print(tfidf_df.head())
