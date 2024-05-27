from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

def __lstm__(stock_df):
    # fig, ax = plt.subplots(figsize=(20, 10))
    # self.stock_df[["Adj Close", "pred_timesfm"]].plot(ax=ax)
    # st.pyplot(fig)
    
    # --------------------------------------- LSTM -----------------------------------

    data = stock_df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .80 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60 : i, 0])
        y_train.append(train_data[i, 0])
            
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    # -> (B, 60, 1)
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1), use_bias = False))
    # -> (B, 60, 128)
    model.add(LSTM(64, return_sequences=False, use_bias = False))
    # -> (B, 64)
    model.add(Dense(25))
    # -> (B, 25)
    model.add(Dense(1))
    # -> (B, 1)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    test_data = scaled_data[: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
        
    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    preds = model.predict(x_test)
    preds = scaler.inverse_transform(preds)
    print (preds.shape)
    predictions = np.full((60,1), np.nan)
    
    predictions = np.concatenate((predictions, preds), axis=0)
    print(predictions.shape)
    return predictions

    # -------------------------------- timefm + LSTM ---------------------------------

    # st.pyplot(plot_lstm_timefm_prediction(data = stock_df, lstm_prediction = predictions))