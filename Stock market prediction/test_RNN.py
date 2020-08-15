# import required packages
from tensorflow import keras
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
    
    #Load the model of choice
    model = keras.models.load_model('models/20867324_RNN_model.h5')
    
    #Accesing the training dataset to perform RNN
    test_data = pd.read_csv("data/test_data_RNN.csv")
    test_set = test_data.iloc[:, 0:12].values
    test_labels = test_data.iloc[:,12:13].values

    # Re-shaping the train_set before training
    test_set = test_set.reshape(test_set.shape[0],3,4)

    # Run prediction on the test data and output required plot and loss
    predicted_value= model.predict(test_set)
    
    #Calculating the loss after prediction
    k = mean_absolute_error(test_labels, predicted_value)
    print('=================================================================================')
    print('The mean absolute error of this model is :-',k)
    print('=================================================================================')
    
    print('Refer to the Plot between the predicted and target values in the corresponding Figure file :-')
    print('=================================================================================')
    #Plotiing the predicted and actual opening stock prices to visualize better
    fig = plt.figure(figsize=(20,10))
    plt.plot(predicted_value, color= 'red',)
    plt.plot(test_labels, color = 'blue')
    plt.title('Opening prices of stocks being sold')
    plt.legend(['Predicted opening prices', 'Actual Opening prices'])
    plt.xlabel('Time frame')
    plt.ylabel('Stock Opening prices')
    plt.show()