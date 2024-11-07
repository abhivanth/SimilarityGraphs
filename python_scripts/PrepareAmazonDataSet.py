import json

import pandas as pd


def load_amazon_data():
    reviews = []
    with open('data/Amazon_review_dataset_McAuley 2023/Health_and_Personal_Care.jsonl', 'r') as f:
        for line in f:
            reviews.append(json.loads(line))
    reviews_df = pd.DataFrame(reviews)
    metadata = []
    with open('data/Amazon_review_dataset_McAuley 2023/meta_Health_and_Personal_Care.jsonl', 'r') as f:
        for line in f:
            metadata.append(json.loads(line))
    metadata_df = pd.DataFrame(metadata)
    combined_df = pd.merge(reviews_df, metadata_df, how='left', on='parent_asin')
    combined_df = combined_df.head(5000)
    combined_text = pd.concat([combined_df['title_x'].fillna(''), combined_df['text'].fillna('')], axis=1)
    return combined_text
