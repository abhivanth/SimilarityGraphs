from datasets import load_dataset
from transformers import AutoTokenizer, BertModel
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
model = BertModel.from_pretrained("colbert-ir/colbertv2.0")

ds = load_dataset("allenai/dolma-pes2o-cc-pd", split="train", streaming=True)


def tokenize_batch(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)


streamed_ds = ds.map(tokenize_batch, batched=True, batch_size=500)
embeddings_list = []
for count, example in enumerate(streamed_ds, 1):

    inputs = tokenizer(example['text'], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings_list.append(embeddings.cpu().numpy())
    if count >= 500:
        break

print(np.array(embeddings_list).shape)
