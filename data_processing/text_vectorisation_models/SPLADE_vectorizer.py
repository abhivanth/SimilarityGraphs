from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np

model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)


def vectorize_data(df, text_column, id_column):
    embeddings_list = []
    for _, row in df.iterrows():
        inputs = tokenizer(row[text_column], return_tensors="pt", padding=True, truncation=True)
        batch_id = np.array(row[id_column], dtype=np.float32)
        with torch.no_grad():
            logits = model(**inputs).logits
        embeddings = torch.log(1 + torch.relu(logits)).max(dim=1).values
        embeddings_list.append(np.column_stack((embeddings.cpu().numpy(), batch_id)))
    return np.vstack(embeddings_list)
