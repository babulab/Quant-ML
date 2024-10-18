import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import math

#Function to download the data
def get_data(ticker, start_date, end_date):
    return pd.DataFrame(yf.download(ticker, start=start_date, end=end_date))['Adj Close']

#Time feature
def add_date_features(df_original): 
    df = df_original.copy()
    
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    
    # Cyclic encoding for Month
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Cyclic encoding for Day of the Week
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    # Drop the original non-cyclic columns
    df = df.drop(['Month', 'DayOfWeek'], axis=1)
    
    return df

#Finance features
def compute_sma(data, window): #Simple Moving Average 
    sma_df = pd.DataFrame()
    for col in data.columns:
        sma_df[f'{col}_SMA_{window}'] = data[col].rolling(window=window).mean()
    return sma_df


def compute_ema(data, window): # Exponential Moving Average
    ema_df = pd.DataFrame()
    for col in data.columns:
        ema_df[f'{col}_EMA_{window}'] = data[col].ewm(span=window, adjust=False).mean()
    return ema_df


def compute_rsi(data, window=14): #Relative Strength Index 
    rsi_df = pd.DataFrame()
    for col in data.columns:
        delta = data[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi_df[f'{col}_RSI_{window}'] = 100 - (100 / (1 + rs))
    return rsi_df


def compute_macd(data, fast=12, slow=26, signal=9): #Moving Average Convergence Divergence 
    macd_df = pd.DataFrame()
    signal_df = pd.DataFrame()
    for col in data.columns:
        ema_fast = data[col].ewm(span=fast, adjust=False).mean()
        ema_slow = data[col].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_df[f'{col}_MACD'] = macd_line
        signal_df[f'{col}_MACD_Signal'] = signal_line
    return macd_df, signal_df


def compute_bollinger_bands(data, window=20): #Volatility (Standard Deviation or Bollinger Bands)
    upper_band_df = pd.DataFrame()
    lower_band_df = pd.DataFrame()
    for col in data.columns:
        sma = data[col].rolling(window=window).mean()
        stddev = data[col].rolling(window=window).std()
        upper_band_df[f'{col}_BB_Upper'] = sma + (stddev * 2)
        lower_band_df[f'{col}_BB_Lower'] = sma - (stddev * 2)
    return upper_band_df, lower_band_df




def preprocess_data(df_data, name_target, name_features, window_size, scaler='MinMax'):

    data = df_data.copy()

    X_data = data[name_features]
    y_data = data[name_target]
    idx_target = name_features.index(name_target)


    # Select scaler based on the input scaler
    if scaler == 'MinMax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(X_data)
    elif scaler == 'StandardScaler':
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X_data)
    elif scaler =='Without':
        print('No hay scaling')
        
        scaled_data = X_data.values
        
    else:
        # Transform using the already fitted scaler
        scaled_data = scaler.transform(X_data)


    X, y, info_dates = [], [], {'X':[], 'y':[]}
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i, idx_target])
        info_dates['X'].append((X_data[i-window_size:i]).index)
        info_dates['y'].append(y_data.index[i])

    X, y = np.array(X), np.array(y) 

    return X, y, info_dates, scaler, idx_target


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # reset counter if validation loss improves
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def make_predictions(model, X_loader):
    model.eval()
    with torch.no_grad():
        predictions = model(X_loader)
    return predictions.detach()                

#Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
    

class modelTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, output_size=1,max_len=60):
        super(modelTransformer, self).__init__()
        self.d_model = d_model
        
        #d_model =embed_dim
        # Linear transformation before the transformer
        self.input_fc = nn.Linear(input_size, d_model)

        # Transformer encoder
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Linear output layer
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = x.transpose(0,1)

        #x = self.transformer(x,x)
        x = self.transformer_encoder(x)

        output = self.fc_out(x[-1,:, :]) 

        return output.squeeze()


#LSTM
class modelLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=1, output_size=1):
        super(modelLSTM, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1])
        return predictions.view(-1) 
    


#plots

def plot_results(real, train_real, val_real, test_real, train_pred, val_pred, test_pred):
    # Plotting predicted values for train, val, and test sets
    
    plt.figure(figsize=(14, 8), dpi=120)    
    # Plotting real values
    real = real.copy()
    name_target = real.name
    real_combined = pd.DataFrame(real)
    real_combined['Date'] = pd.to_datetime(real_combined.index)
    real_combined.reset_index(inplace=True, drop=True)

    date_diff = real_combined['Date'].diff()
    break_indices = np.where(date_diff > timedelta(days=5))[0]

    start_idx = 0
    for end_idx in break_indices:
        plt.plot(real_combined['Date'][start_idx:end_idx], real_combined[name_target][start_idx:end_idx], color='k')
        start_idx = end_idx
    plt.plot(real_combined['Date'][start_idx:], real_combined[name_target][start_idx:], color='k', label='Real Stock Price')

    
    plt.plot(train_real.index, train_pred, color='C0', label='Train Prediction')
    plt.plot(val_real.index, val_pred, color='C1', label='Validation Prediction')
    plt.plot(test_real.index, test_pred, color='C2', label='Test Prediction')

    plt.title('Stock ' + train_real.name +' Price Prediction', size=14)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


def plot_metrics(losses, metrics):
    fig, ax = plt.subplots(nrows=3,figsize=(10, 7), dpi=120)

    plt.suptitle('Metrics during the training', size=14)
    ax[0].plot(losses['train'], label='Train Loss')
    ax[0].plot(losses['val'], label='Validation Loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss (MSE)')
    ax[0].set_yscale('log')
    ax[0].legend(loc=1)

    ax[1].plot(metrics['mape_train'], label='Train MAPE')
    ax[1].plot(metrics['mape_val'], label='Validation MAPE')
    ax[1].set_title('MAPE')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('MAPE')
    ax[1].legend(loc=1)

    ax[2].plot(metrics['rmse_train'], label='Train RMSE')
    ax[2].plot(metrics['rmse_val'], label='Validation RMSE')
    ax[2].set_title('RMSE')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('RMSE')
    ax[2].legend(loc=1)

    plt.subplots_adjust(hspace=0.7)
    plt.show()