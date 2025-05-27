from python_scripts.Save_Stream_Data import save_stream_to_parquet
import pandas as pd

ds_filename = save_stream_to_parquet("pmc/open_access", num_records=500)
df_pmc = pd.read_parquet(ds_filename)
