import pandas as pd
import time
from geobleu import geobleu

pd.options.mode.chained_assignment = None

def convert_label_back(df):
    '''
    Convert label back to x, y
    '''
    df['predict_x'] = (df['label'] // 200) + 1
    df['predict_y'] = (df['label'] % 200) + 1
    return df

def get_time_str():
    return time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))

def load_data(city='A'):
    '''
    Load data from csv file
    '''
    if city == 'A':
        df = pd.read_csv('your_path_here/cityA_groundtruthdata.csv.gz', compression='gzip')
    else:
        df = pd.read_csv(f'your_path_here/city{city}_challengedata.csv.gz', compression='gzip')
    
    users = sorted(list(df['uid'].unique()))
    predict_users = users[-3000:]
    train_df = df[~df['uid'].isin(predict_users)]
    predict_df = df[df['uid'].isin(predict_users)]
    return train_df, predict_df

def calc_bleu_dtw_loss(generated, target):
    '''
    Calculate BLEU and DTW loss
    tuple format: (uid, d, t, x, y) or (d, t, x, y)
    '''
    assert len(generated) == len(target)
    geo_bleu = geobleu.calc_geobleu(generated, target, processes=3)
    dtw = geobleu.calc_dtw(generated, target, processes=3)
    accuracy = sum([1 for i in range(len(generated)) if generated[i] == target[i]]) / len(generated)
    return geo_bleu, dtw, accuracy
