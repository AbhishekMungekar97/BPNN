# import required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model

# Imporgting the dataset
df = pd.read_csv("data/q2_dataset.csv")

# Extrating the main features
training_set = df.iloc[:, 2: 6].values
feature_set = training_set

# Creating the dataset such that we have 12 features per sample 
time_step= 3
X=[]
y=[]
for i in range(len(feature_set)-time_step-1):
    t=[]
    for j in range(0,time_step):
        
        t.append(feature_set[[(i+j)], :])
    X.append(t)
    y.append(feature_set[i+ time_step,1])
X = np.array(X)
y= np.array(y)    
    
# Splitting the dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshaping the data before storing it, since we require 12 features per sample
X_test = X_test.reshape(X_test.shape[0],12)
X_train = X_train.reshape(X_train.shape[0],12)

# Pre-processing the data, we use min-max scalar
std_scale = MinMaxScaler().fit(X_train,y_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)

# Adding the targets to the training and testing datasets respectively
h = pd.DataFrame(X_train)
h['target']= y_train

m = pd.DataFrame(X_test)
m['target']= y_test

# Saving the created trained and test dataset as csv files
c = np.savetxt('data/test_data_RNN.csv', m, delimiter =', ')  
d = np.savetxt('data/train_data_RNN.csv', h, delimiter =', ')



if __name__ == "__main__":
    
    print('=================================================================')
    print('Loading the dataset to train.....................................')
    print('=================================================================')
    
    #Accesing the training dataset to perform RNN
    train_data = pd.read_csv("data/train_data_RNN.csv")
    train_set = train_data.iloc[:, 0:12].values
    train_labels = train_data.iloc[:,12:13].values
    
    # Re-shaping the train_set before training
    train_set = train_set.reshape(train_set.shape[0],3,4)
    
    # Training the model using 50 sets of hidden layers
    model = Sequential()
    model.add(LSTM(units=50, return_sequences= True, input_shape=(train_set.shape[1],4)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=30))
    model.add(Dense(units=1))
    model.summary()
    
    # Compiling using an optimizer and approporiate loss function
    model.compile(optimizer='adam', loss='mae')
    print('=================================================================')
    print('Traing the model.................................................')
    print('=================================================================')
    model.fit(train_set, train_labels, epochs=700, batch_size=15)
    
    #Saving the model
    print('=================================================================')
    print('Saving the trained model.........................................')
    print('=================================================================')
    model.save('models/20867324_RNN_model.h5')