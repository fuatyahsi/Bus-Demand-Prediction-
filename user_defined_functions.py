import pandas as pd      
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
from ipywidgets import interact
from ipywidgets import widgets
from prophet import Prophet
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,BatchNormalization,GRU,SimpleRNN
from tensorflow.keras.optimizers import Adam



import warnings
warnings.filterwarnings('ignore')
#plt.rcParams["figure.figsize"] = (7,4)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 700)

######################################################################################################################################################################################################        
    
def is_stationary(dataset):
    ''' Tests whether a time series variable is non-stationary and possesses a unit root by using addfuller test'''
    # H0 : Non_stationary
    # H1 : Stationary 
    if (adfuller(dataset.usage)[1]) < 0.05 :
        print("p_value",(adfuller(dataset.usage)[1]),"Stationary")
    else : 
        print("p_value",(adfuller(dataset.usage)[1]),"Non-Stationary")
        

###################################################################################################################################################################################################### 

def prophet_predict(Municipality,Municipality_train,Municipality_test):
    
    '''Predicts usage values using prophet model for given future dates then evaluates the results by visualizing '''
    # Preparing dataset for prophet model
    M0_prophet = Municipality_train.reset_index()[["timestamp","usage"]]
    M0_prophet.columns=["ds","y"]
    
    #Fitting the model
    model = Prophet()
    model.fit(M0_prophet)

    #Precidicting values
    future = pd.DataFrame(data = Municipality_test.index)
    future.columns = ["ds"]
    forecast_usage = model.predict(future)
    pred = forecast_usage['yhat']
    compare_df = pd.DataFrame({"predictions":pred, "actual":Municipality_test.usage.values})

    #Visualizing actual and predicted values by plotting
    plt.figure(figsize=(7,4))
    sns.lineplot(x = Municipality.index ,y = Municipality.usage)
    sns.lineplot(x = Municipality_test.index ,y = compare_df.predictions,palette ="flare")
    plt.title("ACTUAL AND PREDICTED USAGE VALUES FOR MUNICIPALTY {} by Prohpet".format(int(Municipality_test.municipality_id.mean())),color = "darkred")
    
    #Saving figure
    plt.savefig("C:\\Users\\fuat.yahsi\\Desktop\\github\\Bus-Demand-Prediction-\\Graphs\\USAGE VALUES FOR MUNICIPALTY {} by Prohpet.jpg".format(int(Municipality_test.municipality_id.mean())),dpi = 200)
    
    #Evaulating the model success
    mea = mean_absolute_error(Municipality_test.usage.values,pred)
    global rmse
    rmse = np.sqrt(mean_squared_error(Municipality_test.usage.values,pred))
    print("MEA:",mea,"\nRMSE:",rmse)
    return (pd.DataFrame(data = compare_df.predictions.apply(lambda x : round(x)).values,index = Municipality_test.index,columns = ["Usage Predictions"]).T)

##################################################################################################################################################################################################

def ARIMA_prediction(municipality,municipality_train,municipality_test):
    '''Predicts usage values using prophet model for given future dates then evaluates the results by visualizing '''
    
    #find the order and seasonal order parametres
    parametres = auto_arima(municipality_train.usage, n_jobs=-1,seasonal=True,max_p=20,max_d=20,max_q=20,m = 10) #stationary=True,
    order = parametres.order
    seasonal = parametres.seasonal_order
    
    #create the model and fit 
    model = ARIMA(municipality_train["usage"],order=order,seasonal_order=seasonal,concentrate_scale=True,enforce_invertibility=True,enforce_stationarity=True).fit()
    
    #make predictions
    prediction = model.predict(start=470,end =619) 
    municipality_pred = pd.DataFrame(data = prediction.values,index = municipality_test.index,columns=["prediction"])
    
    #visualize actual and prediction values on lineplot
    plt.figure(figsize=(7,4))
    sns.lineplot(municipality.drop("municipality_id",axis = 1))
    sns.lineplot(municipality_pred,palette ="flare")
    plt.title("ARIMA{}{} results for Municipality {}".format(order,seasonal,municipality.municipality_id.max()),color="darkred",loc = "center")
    
    #Saving figure
    plt.savefig("C:\\Users\\fuat.yahsi\\Desktop\\github\\Bus-Demand-Prediction-\\Graphs\\USAGE VALUES FOR MUNICIPALTY {} by ARIMA.jpg".format(int(municipality_test.municipality_id.mean())),dpi = 200)
    mea = mean_absolute_error(municipality_test.usage.values,prediction)
    global rmse_arima
    rmse_arima = np.sqrt(mean_squared_error(municipality_test.usage.values,prediction))
   
    print("MEA:",mea,"\nRMSE:",rmse_arima)
    return (pd.DataFrame(data = municipality_pred.prediction.apply(lambda x : round(x)).values,index = municipality_pred.index,columns = ["Usage Predictions"]).T)

########################################################################################################################################################

def XGB_predictor(Municipality,train_data,test_data):
    '''Predicts usage values using XGBoostRegressor model for given future dates then evaluates the results by visualizing '''
    #train data
    train_data.drop("municipality_id",axis=1,inplace=True)
    train_data["Month"] = train_data.index.month
    train_data["Day"] = train_data.index.day
    train_data["Day_of_week"] = (train_data.index.day_of_week)+1
    train_data["Hour"] = train_data.index.hour
    train_data = train_data.reset_index()
    train_data.drop("timestamp",axis=1,inplace=True)
    train_data = train_data.iloc[:,[1,2,3,4,0]]
    X_train_xg = train_data.drop("usage",axis = 1)
    y_train_xg = train_data.usage    
    
    
    #test data
    test_data.drop("municipality_id",axis=1,inplace=True)
    test_data["Month"] = test_data.index.month
    test_data["Day"] = test_data.index.day
    test_data["Day_of_week"] = (test_data.index.day_of_week)+1
    test_data["Hour"] = test_data.index.hour
    index = test_data.index
    test_data = test_data.reset_index()
    test_data.drop("timestamp",axis=1,inplace=True)
    test_data = test_data.iloc[:,[1,2,3,4,0]]
    X_test_xg = test_data.drop("usage",axis = 1)
    y_test_xg = test_data.usage  
    
    #model XGBoost Regressor
    model = XGBRegressor(n_estimators=1000, max_depth=13 , max_leaves  = 0)  # found parametres found after hypertparameter-tunning
    model.fit(X=X_train_xg , y=y_train_xg)

    
    #Predictions
    prediction = pd.DataFrame({"Prediction":model.predict(X_test_xg),"Actual":y_test_xg})
    
    
    #Visualizing actual and predicted values together to compare
    plt.figure(figsize=(7,4))
    sns.lineplot(Municipality.drop("municipality_id",axis = 1))
    sns.lineplot(x = index,y = prediction.Prediction.values,palette ="flare")    
    plt.title("ACTUAL and PREDICTION VALUES FOR MUNICIPALITY {} by XGB".format(int(Municipality.municipality_id.mean())),color = "darkred");
    
    #Saving figure
    plt.savefig("C:\\Users\\fuat.yahsi\\Desktop\\github\\Bus-Demand-Prediction-\\Graphs\\USAGE VALUES FOR MUNICIPALTY {} by XGB.jpg".format(int(Municipality.municipality_id.mean())),dpi = 200)
    
    MEA = mean_absolute_error(prediction.Actual,prediction.Prediction)
    global rmse_xgb
    rmse_xgb = np.sqrt(mean_squared_error(prediction.Actual,prediction.Prediction))
    
    print("*"*20)
    print("MEA",MEA)
    print("RMSE",rmse_xgb)
    print("*"*20)
    return (pd.DataFrame(data = prediction.Prediction.apply(lambda x : round(x)).values,index = index,columns = ["Usage Prediction"]).T)
        
    
########################################################################################################################################################

def RNN_predictor(Municipality,Municipality_train,Municipality_test):
    '''Predicts usage values using RNN model for given future dates then evaluates the results by RMSE ans visualizing '''
    # Creating train datasets
    X_train,y_train  = [],[]
    for i in range(10,620):
        X_train.append(Municipality_train.usage.values[i-10:i])
        y_train.append(Municipality_train.usage.values[i])
    X_train,y_train = np.array(X_train),np.array(y_train)


    # Creating a dataset for test (the model's not seen by model before)
    X_test = []
    y_test = []
    for i in range(10,150):
        X_test.append(Municipality_test.usage.values[i-10:i])
        y_test.append(Municipality_test.usage.values[i])
    X_test,y_test = np.array(X_test),np.array(y_test)


    #Scaling and reshaping datasets
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_X_train = scaler_x.fit_transform(X_train)
    scaled_X_test = scaler_x.transform(X_test)
    y_train = y_train.reshape(610,1)
    y_test = y_test.reshape(140,1)
    scaled_y_train = scaler_y.fit_transform(y_train)
    scaled_y_test = scaler_y.transform(y_test)
    scaled_X_train = scaled_X_train.reshape(610,10,1)
    scaled_X_test = scaled_X_test.reshape(140,10,1)
    
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(10, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
        
    
    
    # Compiling model 
    optimizer = Adam(learning_rate= (0.0015))
    model.compile(optimizer=optimizer,loss="mse")

    # Fitting the model
    history = model.fit(scaled_X_train,scaled_y_train,epochs=100,batch_size = 32,validation_split=0.01)

    # Number of parametres and layers
    model.summary()   

    # Visualizing training 
    pd.DataFrame(history.history).plot()


    # Predictions by using last ten values with RNN
    scaled_predictions= model.predict(scaled_X_test)
    preds=scaler_y.inverse_transform(scaled_predictions)

    prediction = pd.DataFrame(data = preds, index = Municipality_test.index[10:], columns=["predictions"])
    prediction["actual"] = Municipality_test[10:].usage.values



    #Visualizing actual and predicted values together to compare
    plt.figure(figsize=(7,4))
    sns.lineplot(data = Municipality.drop("municipality_id",axis = 1))
    sns.lineplot(data = prediction.drop("actual",axis = 1),palette = "YlOrBr")
    plt.legend(loc=(1,0.8))
    plt.title("ACTUAL and PREDICTION VALUES FOR MUNICIPALITY {} by RNN".format(Municipality.municipality_id.max()),color = "darkred");


    #Saving figure
    plt.savefig("C:\\Users\\fuat.yahsi\\Desktop\\github\\Bus-Demand-Prediction-\\Graphs\\USAGE VALUES FOR MUNICIPALTY {} by RNN.jpg".format(Municipality.municipality_id.max()),dpi = 200)

    # Evalualing model with metrics R2 and MSE
    MAE = mean_absolute_error(prediction.actual,prediction.predictions)
   

    global rmse_rnn
    rmse_rnn = np.sqrt(mean_squared_error(prediction.actual,prediction.predictions))

    print("*"*20)
    print("MAE",MAE)
    print("RMSE total",rmse_rnn)
    print("*"*20)
    print("PREDICTED VALUES")

    # Predicted values 
    return (pd.DataFrame({"Usage Prediction":prediction.predictions.apply(lambda x : int(x))},index = Municipality_test.index[10:]).T)