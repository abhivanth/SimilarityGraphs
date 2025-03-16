from datasets import load_dataset
from transformers import AutoTokenizer, BertModel
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
model = BertModel.from_pretrained("colbert-ir/colbertv2.0")

ds = load_dataset("allenai/dolma-pes2o-cc-pd", split="train", streaming=True)

streamed_ds = ds.map(batched=True, batch_size=500)
embeddings_list = []
for count, batch in enumerate(streamed_ds, 1):

    inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
    batch_id = batch['id']
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings_list.append(np.column_stack((embeddings.cpu().numpy(), batch_id)))
    if count >= 500:
        break

final_embeddings = np.vstack(embeddings_list)
print(final_embeddings.shape)

np.savetxt("s2orc_colbert_embeddings.csv", final_embeddings, delimiter=",")
