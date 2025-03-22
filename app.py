import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('/content/Churn_Modelling.csv')

df.head()

# preprocess the data
# drop irrelevent features
df=df.drop(['RowNumber','CustomerId','Surname'],axis=1)

df.head()

# applying label encoder to gender column since it has only 2 values male nd female
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])

df

# applying ohe on geography column
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
geo_encoder=ohe.fit_transform(df[['Geography']])

geo_encoder

ohe.get_feature_names_out(['Geography'])

geo_encoded_df=pd.DataFrame(geo_encoder,columns=ohe.get_feature_names_out(['Geography']))
geo_encoded_df

# combine geography column with data

df=pd.concat([df.drop(['Geography'],axis=1),geo_encoded_df],axis=1)

df.head()

# save the encoders and scalers for future use
with open('label_encoder.pkl','wb') as f:
  pickle.dump(le,f)

with open('one_hot_encoder.pkl','wb') as f:
  pickle.dump(ohe,f)

# divide the dataset into independent and dependent features
df.head()
x=df.drop(['Exited'],axis=1)
y=df['Exited']

# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# scale the data

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

x_train

# save the scaler for future use
with open('scaler.pkl','wb') as f:
  pickle.dump(sc,f)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense  #-used to create neurons in hidden layers
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

#build our ANN model

model=Sequential([
    Dense(64,activation='relu',input_shape=(x_train.shape[1],)),  # HL1 connected to input layer, x_train.shape[1] will give no. of colums we have
    Dense(32,activation='relu'),    # HL2
    Dense(1,activation='sigmoid')])  # output layer ,used sigmoid bcz only 1 output i.e binary classification problem


# yaha se humne ye krlia ki humare pas 12 inputs hai jo HL1 jisme 64 neurons set kiye humne unse connected h then vo further HL2 jisme
# 32 neurons set kiye unse connected h and finally vo 1 output layer ke neuron se connected h


model.summary()

# params are total no. of combinations of weights and bias

# compile the model for forward and backward proposition

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])  #we chose binary cross entropy as we have binary classification problem

# setup the tensorboard     # tensorboard is used to visualize all the logs that we have while training the model

from tensorflow.keras.callbacks import EarlyStopping,TensorBoard

log_dir='logs/fit/'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

# setup early stopping - used bcz suppose model is performing 50 epochs but we see that after performing 20 epochs model has been trained
# at its best level and after that it is improving by very very small rate so no need to train it further
# and also if after some epochs loss value is not decreasing further then also apply this early stopping

early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,mode='max',restore_best_weights=True)


# train the model

history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,callbacks=[tensorboard_callback,early_stopping_callback])

model.save('churn_model.h5')

# load tensorboard extension

#%load_ext tensorboard

%tensorboard --logdir logs/fit

# load the pickle file
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

# load the ANN trained model,scaler pickle file,onehot pickle file

model=load_model('/content/churn_model.h5')

with open('/content/scaler.pkl','rb') as f:
  sc=pickle.load(f)

with open('/content/one_hot_encoder.pkl','rb') as f:
  ohe=pickle.load(f)

with open('/content/label_encoder.pkl','rb') as f:
  le=pickle.load(f)

# example input data

input_data={
    'CreditScore':600,
    'Geography':'France',
    'Gender':'Male',
    'Age':42,
    'Tenure':3,
    'Balance':100000,
    'NumOfProducts':2,
    'HasCrCard':1,
    'IsActiveMember':1,
    'EstimatedSalary':50000
}


# One hot encode 'geography'

geo_encoded=ohe.transform([[input_data['Geography']]])
geo_encoded_df=pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out(['Geography']))
geo_encoded_df

input_df=pd.DataFrame([input_data])
input_df

# label encode gender

input_df['Gender']=le.transform(input_df['Gender'])

input_df

# concatination one hot encoded data to input_df ,no need to concat gender as its encoding has been done in input_df only

input_df=pd.concat([input_df.drop(['Geography'],axis=1),geo_encoded_df],axis=1)
input_df

#scaling the input data

# Get the column names from when the scaler was fitted
original_columns=sc.feature_names_in_

# Ensure input_df has the same columns and order
input_df=input_df.reindex(columns=original_columns,fill_value=0)

# Now you can scale the input data
input_df=sc.transform(input_df)


input_df

# predict churn

prediction=model.predict(input_df)
prediction

prediction_probability=prediction[0][0]

prediction_probability

if prediction_probability>0.5:
  print('Customer will leave the bank')
else:
  print('Customer will not leave the bank')

!pip install streamlit
import streamlit as st

# load the trained model
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st

model=tf.keras.models.load_model('/content/churn_model.h5')

# load the encoder and scaler
with open('/content/scaler.pkl','rb') as f:
  sc=pickle.load(f)

with open('/content/one_hot_encoder.pkl','rb') as f:
  ohe=pickle.load(f)

with open('/content/label_encoder.pkl','rb') as f:
  le=pickle.load(f)

#streamlit app

st.title('Customer Churn Prediction')

# user input

geography=st.selectbox('Geography',ohe.categories_[0])
gender=st.selectbox('Gender',le.classes_)
age=st.slider('Age',min_value=18,max_value=100)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_Salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',min_value=0,max_value=10)
num_of_Products=st.slider('Number of Products',min_value=1,max_value=4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

# prepare the input data

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[le.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_Products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_Salary],
})

# one hot encode geography

geo_encoded=ohe.transform([[geography]])
geo_encoded_df=pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out(['Geography']))


# combine one hot encoded values with input data

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)



# Get the column names from when the scaler was fitted
original_columns=sc.feature_names_in_

# Ensure input_df has the same columns and order
input_data=input_data.reindex(columns=original_columns,fill_value=0)

# Now you can scale the input data
input_data=sc.transform(input_data)


#prediction churn

prediction=model.predict(input_data)
prediction_probability=prediction[0][0]

if prediction_probability>0.5:
  st.write('Customer will leave the bank')
else:
  st.write('Customer will not leave the bank')

