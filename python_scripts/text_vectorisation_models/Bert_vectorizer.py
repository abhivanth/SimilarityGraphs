from transformers import AutoTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

tokenizer_colbert = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
model_colbert = BertModel.from_pretrained("colbert-ir/colbertv2.0")

bert = 'distilbert-base-uncased'
tokenizer_bert = DistilBertTokenizer.from_pretrained(bert)
model_bert = DistilBertModel.from_pretrained(bert)


def vectorize_data(df, text_column, id_column, model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vectorize_data] Using device: {device}")

    if model == "Bert":
        tokenizer = tokenizer_bert
        model = model_bert.to(device)
    else:
        tokenizer = tokenizer_colbert
        model = model_colbert.to(device)

    embeddings_list = []
    for _, row in df.iterrows():
        inputs = tokenizer(row[text_column], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        batch_id = np.array(row[id_column], dtype=np.float32)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings_list.append(np.column_stack((embeddings.cpu().numpy(), batch_id)))
    return np.vstack(embeddings_list)
