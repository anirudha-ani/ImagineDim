# import required modules
import pandas as pd


def loadVerifiedShapenetData(dataPath, labelsToFetch: list):
    chunk = pd.read_csv(dataPath, chunksize=1000)
    
    shapenet_df = pd.concat(chunk)
    # print(shapenet_df.columns.values)

    return shapenet_df[labelsToFetch]