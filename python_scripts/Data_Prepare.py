from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json

with open("data/News_Category_Dataset_v3.json/News_Category_Dataset_v3.json", "r") as news_data_set_json:
    news_data_set = json.load(news_data_set_json)

# Extract categories and descriptions
news_data_set = news_data_set["data"]
categories = np.array([item["category"] for item in news_data_set])
descriptions = [item["short_description"] for item in news_data_set]

tf_idf = TfidfVectorizer(max_features=5)
vector_description = tf_idf.fit_transform(descriptions).toarray()

full_dataset = np.column_stack((vector_description,categories))
np.savetxt("newsdata_set.csv", full_dataset, delimiter=",",fmt="%s")
