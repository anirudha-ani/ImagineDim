import spacy
import pandas as pd
import numpy as np
import gensim
import enchant
import math

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
     embedding_model = gensim.models.KeyedVectors.load_word2vec_format('Dataset/GoogleNews/GoogleNews-vectors-negative300.bin', binary=True)

     # for index in range(len(data)): 
     #      if()

     for index in range(len(data)): 
          if(type(data[index][0]) != str or not(isinstance(data[index][1], (int, float))) or np.isnan(data[index][1])):
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
                    if len(token.text)> 2 and eng_dict.check(token.text) and (token.text in embedding_model):
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
     input_data_embedded_form = embedding_model[input_data]
     
     no_of_total_data = np.shape(label)[0]
     train_split = 0.9
     no_of_train_data = int(math.ceil(no_of_total_data * train_split))

     shuffled_index = np.arange(no_of_total_data)
     np.random.shuffle(shuffled_index)
     
     # print(np.shape(input_data_embedded_form))
     # print(np.shape(label))

     shuffled_input_data = np.take(input_data_embedded_form, shuffled_index, axis=0)
     shuffled_english_input_word = np.take(input_data, shuffled_index, axis = 0)
     shuffled_label_data = np.take(label, shuffled_index, axis = 0)

     # print(np.shape(shuffled_input_data))
     # print(np.shape(shuffled_label_data))

     train_x = shuffled_input_data[0:no_of_train_data, :]
     train_english_word = shuffled_english_input_word[0:no_of_train_data]
     train_y = shuffled_label_data[0:no_of_train_data]

     test_x = shuffled_input_data[no_of_train_data: no_of_total_data, :]
     test_english_word = shuffled_english_input_word[no_of_train_data: no_of_total_data]
     test_y = shuffled_label_data[no_of_train_data: no_of_total_data]
     # print(np.shape(input_data_embedded_form))
     # print(np.shape(label))
     # print(np.shape(train_x))
     # print(np.shape(train_y))
     # print(np.shape(test_x))
     # print(np.shape(test_y))

     return train_x, train_y, test_x, test_y, train_english_word, test_english_word

     # return data[:, 0]
     # doc = nlp(input_text)

get_shapenet_data()
