from transformers import AutoTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

tokenizer_colbert = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
model_colbert = BertModel.from_pretrained("colbert-ir/colbertv2.0")

bert = 'distilbert-base-uncased'
tokenizer_bert = DistilBertTokenizer.from_pretrained(bert)
model_bert = DistilBertModel.from_pretrained(bert)


def vectorize_data(df, text_column, id_column, model=None):
    if model == "Bert":
        tokenizer = tokenizer_bert
        model = model_bert
    else:
        tokenizer = tokenizer_colbert
        model = model_colbert

    embeddings_list = []
    for _, row in df.iterrows():
        inputs = tokenizer(row[text_column], return_tensors="pt", padding=True, truncation=True)
        batch_id = np.array(row[id_column], dtype=np.float32)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings_list.append(np.column_stack((embeddings.cpu().numpy(), batch_id)))
    return np.vstack(embeddings_list)
