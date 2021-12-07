import spacy
import pandas as pd
import numpy as np
import gensim
import enchant

# take shapenetdata and return name of the object and volume
def get_shapenet_data():

     # reading the shapenet data from CSV
     chunk = pd.read_csv('Dataset/ShapeNet/verified_shapenet_data.csv', chunksize=1000)
    
     shapenet_df = pd.concat(chunk)
     # print(shapenet_df.columns.values)

     # only taking name and solidVolume column from the dataset
     dataFrame = shapenet_df[ ['name','solidVolume']]

     # f = open("output.txt", "a")
     # print("Hello stackoverflow!", file=f)
     # print("I have a question.", file=f)
     # f.close()
     
     data = dataFrame.to_numpy()
     # print(np.shape(data))
     # print(data[:,0])


     # Below code changes the name from multiple name to a single noun
     nlp = spacy.load("en_core_web_sm")
     unprocessed_data_index = [] 
     input_data = []
     label = []
     eng_dict = enchant.Dict("en_US")
     model = gensim.models.KeyedVectors.load_word2vec_format('Dataset/GoogleNews/GoogleNews-vectors-negative300.bin', binary=True)

     for index in range(len(data)): 
          if(type(data[index][0]) != str or not(isinstance(data[index][1], (int, float)))):
               # print("Ignoring = ", data[index][0])
               unprocessed_data_index.append(index)
               continue
          # print(data[index][0], " changing into = ",file=f)
          doc = nlp(data[index][0])
          
          nounPhrases = []
          lastToken = None
          for token in doc:
               lastToken = token.text
               if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                    # print("Subject = ", token.text)
                    if eng_dict.check(token.text) and (token.text in model):
                         nounPhrases.append(token.text)
          if(len(nounPhrases) != 0):
               input_data.append(nounPhrases.pop())
               label.append(data[index][1])
          # else:
          #      if( eng_dict.check(lastToken)):
          #           input_data.append(lastToken)
          #           label.append(data[index][1])
          # print(data[index][0], file=f)
     # print("Unprocessed data index = ", unprocessed_data_index)
     # data = np.delete(data, unprocessed_data_index)
     
     # print(np.shape(data))
     # print(np.shape(input_data))
     # f.close()

     # input_data = data[:, 0]
     # label = data[:, 1]

     # model = gensim.models.KeyedVectors.load_word2vec_format('Dataset/GoogleNews/GoogleNews-vectors-negative300.bin', binary=True)
     input_data_embedded_form = model[input_data]

     # print(np.shape(input_data_embedded_form))
     # print(np.shape(label))

     return input_data_embedded_form, label

     # return data[:, 0]
     # doc = nlp(input_text)

get_shapenet_data()
