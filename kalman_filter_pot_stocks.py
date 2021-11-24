import pandas as pd
import numpy as np
#from arch import arch_model
import datetime as dt
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.stats import norm
from matplotlib.widgets import Slider
import requests


def main():
    data_frame = generate_data()

    run_multi_dimensional(data_frame)

    #print(data_frame)

    #run_one_dimensional(data_frame)

def m_predict(x, P, F, Q):
    return (np.dot(x, F.T) + np.array([x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7], 0, 0, 0, 0]), np.dot(np.dot(F, P), F.T) + Q)

def m_update(x, P, H, R, z, n):
    y = z - np.dot(H, x)
    K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
    return (x + np.dot(K, y), np.dot(np.eye(n) - np.dot(K, H), P))


def o_predict(gaussian):
    return (gaussian[0], gaussian[1] + 2)

def gaussian_multiply(g1, g2):
    mean = (g1[1] * g2[0] + g2[1] * g1[0]) / (g1[1] + g2[1])
    variance = (g1[1] * g2[1]) / (g1[1] + g2[1])
    return (mean, variance)

def o_update(prior, likelihood):
    return gaussian_multiply(prior, likelihood)

def run_one_dimensional(data_frame):
    x = (0, 50)
    xm = []
    xv = []
    z = []

    for i in range(len(data_frame['APHA_close_returns'][1:])):
        prior = o_predict(x)
        likelihood = (data_frame['APHA_close_returns'][i + 1], 30)
        x = o_update(prior, likelihood)
        xm.append(x[0])
        xv.append(x[1])
        xs = np.array([xm, xv])
        q1 = norm.ppf(0.05, xs[0], np.sqrt(xs[1]))
        q2 = norm.ppf(0.95, xs[0], np.sqrt(xs[1]))
        z.append(data_frame['APHA_close_returns'][i + 2])
        plt.plot(z, 'b')
        plt.plot(xs[0], 'k--')
        plt.plot(q1, 'r--')
        plt.plot(q2, 'r--')
        plt.show()

def run_multi_dimensional(data_frame):
    n = 4
    x = np.array([40, 10, 30, 30, 0, 0, 0, 0])

    P = np.diag([100, 100, 100, 100, 5, 5, 5, 5])

    F = np.eye(8)

    Q = np.array([
        [10, 0, 0, 0, 0, 0, 0, 0],
        [0, 10, 0, 0, 0, 0, 0, 0],
        [0, 0, 10, 0, 0, 0, 0, 0],
        [0, 0, 0, 10, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.04, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.04, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.04, 0],
        [0, 0, 0, 0, 0, 0, 0, 0.04]
    ])

    H = np.eye(8)

    R = np.array([
        [50, 0, 0, 0, 0, 0, 0, 0],
        [0, 50, 0, 0, 0, 0, 0, 0],
        [0, 0, 50, 0, 0, 0, 0, 0],
        [0, 0, 0, 50, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0, 0, 5]
    ])

    grwg_price = []
    grwg_variance = []
    grwg_true = []
    grwg_MSE = []

    apha_price = []
    apha_variance = []
    apha_true = []
    apha_MSE = []

    tcnnf_price = []
    tcnnf_variance = []
    tcnnf_true = []
    tcnnf_MSE = []

    cgc_price = []
    cgc_variance = []
    cgc_true = []
    cgc_MSE = []

    for i in range(len(data_frame['GRWG_close'][1:])):
        x, P = m_predict(x, P, F, Q)

        predicted_x = x;

        grwg = data_frame['GRWG_close'][i + 1]
        apha = data_frame['APHA_close'][i + 1]
        tcnnf = data_frame['TCNNF_close'][i + 1]
        cgc = data_frame['CGC_close'][i + 1]
        grwg_returns = data_frame['GRWG_close_returns'][i + 1]
        apha_returns = data_frame['APHA_close_returns'][i + 1]
        tcnnf_returns = data_frame['TCNNF_close_returns'][i + 1]
        cgc_returns = data_frame['CGC_close_returns'][i + 1]

        avg_returns = (grwg_returns + apha_returns + tcnnf_returns + cgc_returns)/4

        z = np.array([grwg, apha, tcnnf, cgc, avg_returns - grwg_returns, avg_returns - apha_returns,  avg_returns - tcnnf_returns, avg_returns - cgc_returns])
        x, P = m_update(x, P, H, R, z, n*2)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex='all', figsize=(16, 9))

        grwg_price.append(predicted_x[0])
        grwg_variance.append(P[0][0])
        q1 = norm.ppf(0.05, grwg_price, np.sqrt(grwg_variance))
        q2 = norm.ppf(0.95, grwg_price, np.sqrt(grwg_variance))
        grwg_true.append(grwg)
        grwg_MSE.append(calculate_MSE(grwg_price, grwg_true))
        ax1.set_title('GRWG Kalman Filter')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Price')
        ax1.plot(grwg_true, 'b')
        ax1.plot(grwg_price, 'k--')
        ax1.plot(q1, 'r--')
        ax1.plot(q2, 'r--')

        apha_price.append(predicted_x[1])
        apha_variance.append(P[1][1])
        q1 = norm.ppf(0.05, apha_price, np.sqrt(apha_variance))
        q2 = norm.ppf(0.95, apha_price, np.sqrt(apha_variance))
        apha_true.append(apha)
        apha_MSE.append(calculate_MSE(apha_price, apha_true))
        ax2.set_title('APHA Kalman Filter')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Price')
        ax2.plot(apha_true, 'b')
        ax2.plot(apha_price, 'k--')
        ax2.plot(q1, 'r--')
        ax2.plot(q2, 'r--')

        tcnnf_price.append(predicted_x[2])
        tcnnf_variance.append(P[2][2])
        q1 = norm.ppf(0.05, tcnnf_price, np.sqrt(tcnnf_variance))
        q2 = norm.ppf(0.95, tcnnf_price, np.sqrt(tcnnf_variance))
        tcnnf_true.append(tcnnf)
        tcnnf_MSE.append(calculate_MSE(tcnnf_price, tcnnf_true))
        ax3.set_title('TCNNF Kalman Filter')
        ax3.set_xlabel('Days')
        ax3.set_ylabel('Price')
        ax3.plot(tcnnf_true, 'b')
        ax3.plot(tcnnf_price, 'k--')
        ax3.plot(q1, 'r--')
        ax3.plot(q2, 'r--')

        cgc_price.append(predicted_x[3])
        cgc_variance.append(P[3][3])
        q1 = norm.ppf(0.05, cgc_price, np.sqrt(cgc_variance))
        q2 = norm.ppf(0.95, cgc_price, np.sqrt(cgc_variance))
        cgc_true.append(cgc)
        cgc_MSE.append(calculate_MSE(cgc_price, cgc_true))
        ax4.set_title('CGC Kalman Filter')
        ax4.set_xlabel('Days')
        ax4.set_ylabel('Price')
        ax4.plot(cgc_true, 'b')
        ax4.plot(cgc_price, 'k--')
        ax4.plot(q1, 'r--')
        ax4.plot(q2, 'r--')

        ax5.set_title('MSE Plot')
        ax5.set_xlabel('Days')
        ax5.set_ylabel('MSE')
        ax5.plot(grwg_MSE, 'b')
        ax5.plot(apha_MSE, 'k')
        ax5.plot(tcnnf_MSE, 'g')
        ax5.plot(cgc_MSE, 'r')

        plt.show()

def calculate_MSE(predict_data, true_data):
    MSE = 0
    for i in range(len(predict_data)):
        MSE += (predict_data[i] - true_data[i]) ** 2
    return MSE

def generate_data():
    api_key = '3b2bdb6cee3045dabbec01da8c4248cf'
    ticker = 'GRWG'
    interval = '1day'
    api_url = f'https://api.twelvedata.com/time_series?symbol={ticker}&start_date=2021-01-01&end_date=2021-03-21&order=ASC&interval={interval}&apikey={api_key}'

    data = requests.get(api_url).json()
    data_frame = pd.DataFrame(data['values'])
    # data_frame.datetime = pd.to_datetime(data_frame.datetime, dayfirst=True)
    # data_frame.set_index("datetime", inplace=True)
    data_frame['open'] = pd.to_numeric(data_frame['open'])
    data_frame[f'{ticker}_open_returns'] = data_frame.open.pct_change(1)
    data_frame[f'{ticker}_open'] = data_frame['open']
    data_frame['close'] = pd.to_numeric(data_frame['close'])
    data_frame[f'{ticker}_close_returns'] = data_frame.close.pct_change(1)
    data_frame[f'{ticker}_close'] = data_frame['close']
    del data_frame['high'], data_frame['low'], data_frame['volume'], data_frame['close'], data_frame['open']

    ticker = 'APHA'
    api_url = f'https://api.twelvedata.com/time_series?symbol={ticker}&start_date=2021-01-01&end_date=2021-03-21&order=ASC&interval={interval}&apikey={api_key}'

    temp = pd.DataFrame(data['values'])
    # temp.datetime = pd.to_datetime(temp.datetime, dayfirst=True)
    # temp.set_index("datetime", inplace=True)
    data_frame['open'] = temp['open']
    data_frame['close'] = temp['close']
    data_frame['open'] = pd.to_numeric(data_frame['open'])
    data_frame[f'{ticker}_open_returns'] = data_frame.open.pct_change(1)
    data_frame[f'{ticker}_open'] = data_frame['open']
    data_frame['close'] = pd.to_numeric(data_frame['close'])
    data_frame[f'{ticker}_close_returns'] = data_frame.close.pct_change(1)
    data_frame[f'{ticker}_close'] = data_frame['close']
    del data_frame['open'], data_frame['close']



    ticker = 'TCNNF'
    api_url = f'https://api.twelvedata.com/time_series?symbol={ticker}&start_date=2021-01-01&end_date=2021-03-21&order=ASC&interval={interval}&apikey={api_key}'

    data = requests.get(api_url).json()
    temp = pd.DataFrame(data['values'])
    # temp.datetime = pd.to_datetime(temp.datetime, dayfirst=True)
    # temp.set_index("datetime", inplace=True)
    data_frame['open'] = temp['open']
    data_frame['close'] = temp['close']
    data_frame['open'] = pd.to_numeric(data_frame['open'])
    data_frame[f'{ticker}_open_returns'] = data_frame.open.pct_change(1)
    data_frame[f'{ticker}_open'] = data_frame['open']
    data_frame['close'] = pd.to_numeric(data_frame['close'])
    data_frame[f'{ticker}_close_returns'] = data_frame.close.pct_change(1)
    data_frame[f'{ticker}_close'] = data_frame['close']
    del data_frame['open'], data_frame['close']



    ticker = 'CGC'
    api_url = f'https://api.twelvedata.com/time_series?symbol={ticker}&start_date=2021-01-01&end_date=2021-03-21&order=ASC&interval={interval}&apikey={api_key}'

    data = requests.get(api_url).json()
    temp = pd.DataFrame(data['values'])
    # temp.datetime = pd.to_datetime(temp.datetime, dayfirst=True)
    # temp.set_index("datetime", inplace=True)
    data_frame['open'] = temp['open']
    data_frame['close'] = temp['close']
    data_frame['open'] = pd.to_numeric(data_frame['open'])
    data_frame[f'{ticker}_open_returns'] = data_frame.open.pct_change(1)
    data_frame[f'{ticker}_open'] = data_frame['open']
    data_frame['close'] = pd.to_numeric(data_frame['close'])
    data_frame[f'{ticker}_close_returns'] = data_frame.close.pct_change(1)
    data_frame[f'{ticker}_close'] = data_frame['close']
    del data_frame['open'], data_frame['close']

    '''

    ticker = 'CRLBF'
    api_url = f'https://api.twelvedata.com/time_series?symbol={ticker}&start_date=2021-01-01&end_date=2021-03-21&order=ASC&interval={interval}&apikey={api_key}'

    data = requests.get(api_url).json()
    temp = pd.DataFrame(data['values'])
    # temp.datetime = pd.to_datetime(temp.datetime, dayfirst=True)
    # temp.set_index("datetime", inplace=True)
    data_frame['open'] = temp['open']
    data_frame['close'] = temp['close']
    data_frame['open'] = pd.to_numeric(data_frame['open'])
    data_frame[f'{ticker}_open_returns'] = data_frame.open.pct_change(1).mul(100)
    data_frame[f'{ticker}_open'] = data_frame['open']
    data_frame['close'] = pd.to_numeric(data_frame['close'])
    data_frame[f'{ticker}_close_returns'] = data_frame.close.pct_change(1).mul(100)
    data_frame[f'{ticker}_close'] = data_frame['close']
    del data_frame['open'], data_frame['close']

    ticker = 'VFF'
    api_url = f'https://api.twelvedata.com/time_series?symbol={ticker}&start_date=2021-01-01&end_date=2021-03-21&order=ASC&interval={interval}&apikey={api_key}'

    data = requests.get(api_url).json()
    temp = pd.DataFrame(data['values'])
    # temp.datetime = pd.to_datetime(temp.datetime, dayfirst=True)
    # temp.set_index("datetime", inplace=True)
    data_frame['open'] = temp['open']
    data_frame['close'] = temp['close']
    data_frame['open'] = pd.to_numeric(data_frame['open'])
    data_frame[f'{ticker}_open_returns'] = data_frame.open.pct_change(1).mul(100)
    data_frame[f'{ticker}_open'] = data_frame['open']
    data_frame['close'] = pd.to_numeric(data_frame['close'])
    data_frame[f'{ticker}_close_returns'] = data_frame.close.pct_change(1).mul(100)
    data_frame[f'{ticker}_close'] = data_frame['close']
    del data_frame['open'], data_frame['close']

    ticker = 'GTBIF'
    api_url = f'https://api.twelvedata.com/time_series?symbol={ticker}&start_date=2021-01-01&end_date=2021-03-21&order=ASC&interval={interval}&apikey={api_key}'

    data = requests.get(api_url).json()
    temp = pd.DataFrame(data['values'])
    # temp.datetime = pd.to_datetime(temp.datetime, dayfirst=True)
    # temp.set_index("datetime", inplace=True)
    data_frame['open'] = temp['open']
    data_frame['close'] = temp['close']
    data_frame['open'] = pd.to_numeric(data_frame['open'])
    data_frame[f'{ticker}_open_returns'] = data_frame.open.pct_change(1).mul(100)
    data_frame[f'{ticker}_open'] = data_frame['open']
    data_frame['close'] = pd.to_numeric(data_frame['close'])
    data_frame[f'{ticker}_close_returns'] = data_frame.close.pct_change(1).mul(100)
    data_frame[f'{ticker}_close'] = data_frame['close']
    del data_frame['open'], data_frame['close']
    '''

    return data_frame


if __name__ == "__main__":
    main()