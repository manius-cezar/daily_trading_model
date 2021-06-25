from typing import Iterable, Tuple
import pandas as pd
import numpy as np
import os
from scipy.stats import kurtosis, skew


def create_features(df: pd.DataFrame, window_length: int, 
        pred_window_length: int, use_old_y: bool = True, ema_lengths: list = [], 
        ema_length_for_mean_model: int = 250) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    avg_stock_price_data = df.loc[:, 'avg_price_in_interval'].to_numpy()
    high_stock_price_data = df.loc[:, 'High'].to_numpy()
    volume_data = df.loc[:, 'Volume'].to_numpy()
    close_data = df['Close']
    # EMA13 calculation
    ema13 = close_data.ewm(span=13, adjust=False).mean().to_numpy()
    ema = df['avg_price_in_interval'].ewm(span=ema_length_for_mean_model, adjust=False).mean().to_numpy()
    
    emas = [ema_calculation(span, close_data) for span in ema_lengths]

    n = len(avg_stock_price_data)
    max_avg_stock_price_in_session = np.max(avg_stock_price_data)
    min_avg_stock_price_in_session = np.min(avg_stock_price_data)
    max_min_diff = max_avg_stock_price_in_session - min_avg_stock_price_in_session

    i = 0

    c = window_length + pred_window_length + 1

    for i in range(0, n - c):
        avg_stock_price_values_current_window = avg_stock_price_data[i:i+window_length]
        avg_stock_price_values_so_far_in_session = avg_stock_price_data[:i+window_length]

        # Features for mean model
        # ======================
        prev_avg_price = avg_stock_price_values_current_window[-2]
        curr_avg_price = avg_stock_price_values_current_window[-1]
        if i == 0:
            prev_ema = ema[window_length + i]
            curr_ema = ema[window_length + i]
        else:
            prev_ema = ema[window_length + i-1]
            curr_ema = ema[window_length + i]
        # ======================

        time_indicator = (i+1)/(n - window_length - pred_window_length)

        mean = np.mean(avg_stock_price_values_so_far_in_session)
        min_avg_stock_price = np.min(avg_stock_price_values_so_far_in_session)
        max_avg_stock_price = np.max(avg_stock_price_values_so_far_in_session)

        volume = volume_data[i+window_length-1]
        kurt = kurtosis(avg_stock_price_values_so_far_in_session)
        skewness = skew(avg_stock_price_values_so_far_in_session)

        # Bulls' power indicator calculation
        bulls_power = high_stock_price_data[i+window_length-1] - ema13[i+window_length-1]
        statistics = [mean, min_avg_stock_price, max_avg_stock_price, volume, kurt, skewness, bulls_power]
        
        for ema_no in range(len(emas)):
            statistics.append(emas[ema_no][i+window_length-1])
        
        statistics.append(time_indicator)
        
        X = np.concatenate([avg_stock_price_values_current_window, statistics])
        X = X.astype(np.float)
        mean_model_features = np.array([prev_avg_price, curr_avg_price, prev_ema, curr_ema])
        
        if use_old_y:
            y = (avg_stock_price_data[i+window_length]-min_avg_stock_price_in_session)/max_min_diff
        else:
            remain_session_min = np.min(avg_stock_price_data[i+window_length:])
            remain_session_max = np.max(avg_stock_price_data[i+window_length:])
            y = 1 if remain_session_max-remain_session_min == 0 else (avg_stock_price_data[i+window_length] - remain_session_min)/(remain_session_max-remain_session_min)
        yield X, y, mean_model_features
        
        
def ema_calculation(span: int, df: pd.DataFrame):
    return df.ewm(span=span, adjust=False).mean().to_numpy()

        
def segmentate_data(file_path: str, window_length: int, pred_window_length: int, 
        use_old_y: bool = True, ema_lengths: list = [], 
        ema_length_for_mean_model: int = 250) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path)

    features: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]] = create_features(df, window_length, pred_window_length, 
        use_old_y = use_old_y, ema_lengths=ema_lengths, ema_length_for_mean_model=ema_length_for_mean_model)
    xs, ys, mean_model_features = zip(*features)

    X = np.array(xs, dtype=np.float)
    y = np.array(ys, dtype=np.float)
    mean_model_features = np.array(mean_model_features, dtype=np.float)

    X = np.expand_dims(X, axis=-1)

    return X, y, mean_model_features


def segmentate_data_from_dir(dir: str, window_length: int, pred_window_length: int, 
        use_old_y: bool = True, ema_lengths: list = [], 
        ema_length_for_mean_model: int = 250):
    files = os.listdir(dir)
    files.sort()

    total_X = []
    total_y = []
    total_mean_model_features = []
    cnt = 1
    for file_path in files:
        if file_path.endswith(".csv"):
            print(f'{cnt}/{len(files)}')
            X, y, mean_model_features = segmentate_data(os.path.join(dir, file_path), window_length, pred_window_length, 
                use_old_y = use_old_y, ema_lengths=ema_lengths, ema_length_for_mean_model=ema_length_for_mean_model)
            # print(y.shape)
            total_X.extend(X)
            total_y.extend(y)
            total_mean_model_features.extend(mean_model_features)
        cnt+=1

    _X = np.array(total_X)
    _y = np.array(total_y)
    _mean_model_features = np.array(total_mean_model_features)
    
    if pred_window_length == 1:
        _y = _y.reshape((_y.shape[0], 1))
    print(f'X shape: {_X.shape}\ny shape: {_y.shape}\nmean_model_features shape: {_mean_model_features.shape}')
    return _X, _y, _mean_model_features


def segmentate_data_from_files(files: list, dir: str, window_length: int, 
        pred_window_length: int, use_old_y: bool = True, ema_lengths: list = [], 
        ema_length_for_mean_model: int = 250):

    total_X = []
    total_y = []
    total_mean_model_features = []
    cnt = 1
    for file_path in files:
        if file_path.endswith(".csv"):
            print(f'{cnt}/{len(files)}')
            X, y, mean_model_features = segmentate_data(os.path.join(dir, file_path), window_length, pred_window_length, 
                use_old_y = use_old_y, ema_lengths=ema_lengths, ema_length_for_mean_model=ema_length_for_mean_model)
            # print(y.shape)
            total_X.extend(X)
            total_y.extend(y)
            total_mean_model_features.extend(mean_model_features)
        cnt+=1

    _X = np.array(total_X)
    _y = np.array(total_y)
    _mean_model_features = np.array(total_mean_model_features)
    
    if pred_window_length == 1:
        _y = _y.reshape((_y.shape[0], 1))
    print(f'X shape: {_X.shape}\ny shape: {_y.shape}\nmean_model_features shape: {_mean_model_features.shape}')
    return _X, _y, _mean_model_features