# import required modules
import pandas as pd
import numpy as np
import time
  
# time taken to read data
# dataset too large for github - so stored in Onedrive
# This dataset is taken from - https://console.cloud.google.com/storage/browser/measures-grounding 
s_time_chunk = time.time()
chunk = pd.read_csv('../../Dataset/DoQ_noun_obj.csv', chunksize=1000, sep='\t', names=['obj', 'head', 'dim', 'mean', 'perc_5', 'perc_25', 'median', 'perc_75',
'perc_95', 'std'])
e_time_chunk = time.time()
  
print("With chunks: ", (e_time_chunk-s_time_chunk), "sec")

pd_df = pd.concat(chunk)
# data
print(pd_df.tail(10))

# filter out only volume data 

# new_df_volume = pd_df[pd_df['dim'] == 'VOLUME']
# new_df_volume.to_csv(r'../../Dataset/volume.csv', index=False, encoding='utf-8')