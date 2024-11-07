import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
from python_scripts.PrepareAmazonDataSet import load_amazon_data

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def text_to_vector(text):
    inputs = tokenizer.encode_plus(text, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state
    sentence_embedding = embeddings.mean(dim=1)
    return sentence_embedding.squeeze().numpy()


combined_text = load_amazon_data()
result_df_title = combined_text['title_x'].apply(text_to_vector)
result_df_text = combined_text['text'].apply(text_to_vector)

result_df_bert = pd.concat([result_df_title, result_df_text], axis=1)

result_df_bert.to_csv('../data/amazon_bert.csv', index=False)
