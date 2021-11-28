# import required modules
import pandas as pd
import numpy as np
import time
  
# time taken to read data
# dataset too large for github - so stored in Onedrive
# This dataset is taken from - https://console.cloud.google.com/storage/browser/measures-grounding 
s_time_chunk = time.time()
chunk = pd.read_csv('../../Dataset/ShapeNet/verified_shapenet_data.csv', chunksize=1000)
e_time_chunk = time.time()
  
print("With chunks: ", (e_time_chunk-s_time_chunk), "sec")

shapenet_df = pd.concat(chunk)
# data
print(shapenet_df.tail(10))
print(shapenet_df.columns.values)
print(shapenet_df[['name','unit','aligned.dims','solidVolume', 'weight']])
# filter out only volume data 

# new_df_volume = pd_df[pd_df['dim'] == 'VOLUME']
# new_df_volume.to_csv(r'../../Dataset/volume.csv', index=False, encoding='utf-8')