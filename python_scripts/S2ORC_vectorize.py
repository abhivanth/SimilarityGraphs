import numpy as np
import pandas as pd
from python_scripts.text_vectorisation_models import Tfidf_vectorizer, SPLADE_vectorizer, Bert_vectorizer, \
    DeepSeek_Vectoriser
from python_scripts.Save_Stream_Data import save_stream_to_parquet

# load dataset with desired number of records
ds_filename = save_stream_to_parquet("allenai/dolma-pes2o-cc-pd", num_records=500)
df_S2ORC = pd.read_parquet(ds_filename)

#Tfidf Embeddings
Tfidf_vectorizer.vectorize(df_S2ORC, "text", "id", name="s2orc")

# Bert Embeddings
final_embeddings_bert = Bert_vectorizer.vectorize_data(df_S2ORC, 'text', 'id', model="Bert")
print(final_embeddings_bert.shape)
np.savetxt("python_scripts/text_embeddings/s2orc_embeddings_bert.csv", final_embeddings_bert, delimiter=",")
print("Bert Done")

# colBert Embeddings
final_embeddings_colbert = Bert_vectorizer.vectorize_data(df_S2ORC, 'text', 'id')
print(final_embeddings_colbert.shape)
np.savetxt("python_scripts/text_embeddings/s2orc_embeddings_colbert.csv", final_embeddings_colbert, delimiter=",")
print("colbert Done")

# Splade Embeddings
final_embeddings_splade = SPLADE_vectorizer.vectorize_data(df_S2ORC, 'text', 'id')
print(final_embeddings_splade.shape)
np.savetxt("python_scripts/text_embeddings/s2orc_embeddings_splade.csv", final_embeddings_colbert, delimiter=",")
print("splade Done")

# DeepSeek Embeddings
final_embeddings_deepseek = DeepSeek_Vectoriser.vectorize_dataframe(df_S2ORC, text_column="text", id_column="id", model_type="qwen", model_variant="1.5b")
print(final_embeddings_deepseek.shape)
np.savetxt("python_scripts/text_embeddings/s2orc_embeddings_deepseek_qwen_1.5b.csv", final_embeddings_colbert, delimiter=",")
print("deepseek Done")
