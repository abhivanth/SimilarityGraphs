import pandas as pd
from data_processing.text_vectorisation_models import Tfidf_vectorizer, SPLADE_vectorizer, Bert_vectorizer, \
    DeepSeek_Vectoriser
import numpy as np
from data_processing.Save_Stream_Data import save_stream_to_parquet

# load dataset with desired number of records
ds_filename = save_stream_to_parquet("TomTBT/pmc_open_access_section", num_records=500, name="commercial")
df_pmc_open_access = pd.read_parquet(ds_filename)

#Tfidf Embeddings
Tfidf_vectorizer.vectorize(df_pmc_open_access, text_column='introduction', id_column='pmid', name="PMC_Open_access")

# Bert Embeddings
final_embeddings_bert = Bert_vectorizer.vectorize_data(df_pmc_open_access, 'introduction', 'pmid', model="Bert")
print(final_embeddings_bert.shape)
np.savetxt("data_processing/embeddings/PMC_open_access_embeddings_bert.csv", final_embeddings_bert, delimiter=",")

# colBert Embeddings
final_embeddings_colbert = Bert_vectorizer.vectorize_data(df_pmc_open_access, 'introduction', 'pmid')
print(final_embeddings_colbert.shape)
np.savetxt("data_processing/embeddings/PMC_open_access_embeddings_colbert.csv", final_embeddings_colbert, delimiter=",")

# Splade Embeddings
final_embeddings_splade = SPLADE_vectorizer.vectorize_data(df_pmc_open_access, 'introduction', 'pmid')
print(final_embeddings_splade.shape)
np.savetxt("data_processing/embeddings/PMC_open_access_embeddings_splade.csv", final_embeddings_colbert, delimiter=",")

# DeepSeek Embeddings
final_embeddings_deepseek = DeepSeek_Vectoriser.vectorize_dataframe(df_pmc_open_access, text_column="introduction", id_column="pmid", model_type="qwen", model_variant="1.5b")
print(final_embeddings_deepseek.shape)
np.savetxt("data_processing/embeddings/PMC_open_access_embeddings_deepseek_qwen_1.5b.csv", final_embeddings_colbert, delimiter=",")
print("deepseek Done")
