from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from rdt.transformers import OneHotEncoder
import tensorflow as tf
import pandas as pd
import numpy as np


#if  condition is false, then target_col is None
class DataTransformer(object):
    def __init__(self, rawdata, cat_cols, num_cols,target_col, condition):   
       self.rawdata = rawdata   #type:Dataframe
       self.cat_cols = cat_cols  #type:list
       self.num_cols = num_cols   #type:list
       self.target_col = target_col
       self.condition = condition
       self.cat_dims = [(self.rawdata[col].nunique()) for col in self.cat_cols]
       self.n_numCols = len(self.num_cols)
       if not self.condition:
         self.cat_dims = self.cat_dims + [2] if self.cat_dims is not None else [2]
    
    def transformData(self):
         #load_data
         X = self.rawdata.loc[:, self.num_cols + self.cat_cols]
         y = self.rawdata.loc[:, self.target_col]
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2020)
         # preprocess data
         num_prep = make_pipeline(SimpleImputer(strategy='mean'))
         cat_prep = make_pipeline(SimpleImputer(strategy='most_frequent'))
         prep = ColumnTransformer([
            ('num', num_prep, self.num_cols),
            ('cat', cat_prep, self.cat_cols)],
            remainder='drop')
         self.scaler = MinMaxScaler()
         X_train_imputed = prep.fit_transform(X_train)
         X_train_nums = self.scaler.fit_transform(X_train_imputed[:, :len(self.num_cols)])
         X_train_trans = np.concatenate([X_train_nums, X_train_imputed[:, len(self.num_cols):]], axis=1)
         y_train_trans = y_train.values
         df_imputed = pd.DataFrame(data=X_train_trans, columns=self.num_cols + self.cat_cols)
         #when condition is False
         if not self.condition:
            df_imputed[self.target_col] = y_train_trans
         column_data_list = []
         column_data_list.append(X_train_trans[:, :len(self.num_cols)])
         self.ohetransformers = {}
         if not self.condition: 
            oheCols = self.cat_cols + [self.target_col]
         else:
            oheCols = self.cat_cols
         for col in oheCols:
            data = df_imputed[[col]]
            ohe = OneHotEncoder()
            ohe.fit(data, col)
            self.ohetransformers[col] = ohe
            oh = ohe.transform(data).to_numpy()
            column_data_list.append(oh)
         X_train_trans_final = np.concatenate(column_data_list, axis=1).astype(float)
         X_tens = tf.convert_to_tensor(X_train_trans_final, dtype=tf.float32)
         if not self.condition:
            y_tens = None
         else:
            y_tens = tf.convert_to_tensor(y_train_trans, dtype=tf.float32)
            y_tens = tf.reshape(y_tens, [-1,1])
         return X_tens, y_tens, X_train_imputed
      
    def getbatchX(self, X_tens, batch_size, batch_idx):
        
        X_batch = X_tens[batch_idx * batch_size:(batch_idx + 1) * batch_size]   

        return X_batch
    def getbatchY(self, y_tens, batch_size, batch_idx):
        if y_tens is not None:
           y_batch = y_tens[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        else:
           y_batch = None  
        return y_batch 

    def inverse_transform(self, X_tens):
         recovered_column_data_list = []
         column_names = []
         i = 0
         st = len(self.num_cols)
         for col in self.ohetransformers.keys():
            ed = st + self.cat_dims[i]
            column_data = X_tens[:, st:ed]
            ohe = self.ohetransformers[col]
            data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
            recovered_column_data = ohe.reverse_transform(data)[col]
            recovered_column_data_list.append(recovered_column_data)
            column_names.append(col)
            st = ed
            i +=1
         recovered_data = np.column_stack(recovered_column_data_list)
         recovered_data_cat = (pd.DataFrame(recovered_data, columns=column_names))

         num_data = self.scaler.inverse_transform(X_tens[:, :self.n_numCols])
         recovered_data_num = pd.DataFrame(data = num_data, columns = self.num_cols)
         recovered_df = pd.concat([recovered_data_num, recovered_data_cat], axis=1)
         return recovered_df

