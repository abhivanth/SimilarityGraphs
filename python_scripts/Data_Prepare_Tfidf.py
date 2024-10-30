import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer


# Load Amazon reviews dataset (raw data)
reviews = []
with open('data/Amazon_review_dataset_McAuley 2023/Health_and_Personal_Care.jsonl', 'r') as f:
    for line in f:
        reviews.append(json.loads(line))
reviews_df = pd.DataFrame(reviews)

# Load Amazon metadata dataset
metadata = []
with open('data/Amazon_review_dataset_McAuley 2023/meta_Health_and_Personal_Care.jsonl', 'r') as f:
    for line in f:
        metadata.append(json.loads(line))
metadata_df = pd.DataFrame(metadata)


combined_df = pd.merge(reviews_df, metadata_df, how='left', on='parent_asin')
combined_df = combined_df.head(5000)


combined_text = combined_df['title_x'].fillna('') + ' ' + combined_df['text'].fillna('') + ' ' + combined_df['main_category'].fillna('')


tfidf_vectorizer = TfidfVectorizer(max_features=2000)  # Limit to 2000 features for richer representation
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=combined_df.index, columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df['rating'] = combined_df['rating']
tfidf_df.to_csv('tfidf_features_with_rating.csv', index=False)


print(combined_df[['text', 'title_x', 'main_category']].head())
print(tfidf_df.head())

