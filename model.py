import tensorflow as tf
import gensim
import numpy as np

from DataPreprocess.ShapeNet.shapenet import loadVerifiedShapenetData

def main():
    # data = loadVerifiedShapenetData('Dataset/ShapeNet/verified_shapenet_data.csv', ['name','unit','aligned.dims','solidVolume', 'weight'])
    # print(data)
    model = gensim.models.KeyedVectors.load_word2vec_format('Dataset/GoogleNews/GoogleNews-vectors-negative300.bin', binary=True)
    king = model['king']
    print(np.shape(king))

if __name__ == '__main__':
    main()