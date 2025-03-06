import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import talib
import requests
import time
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from datetime import datetime
from itertools import compress
from scipy.signal import argrelextrema


#  Функция для загрузки данных ETH/USDT с Binance
def get_binance_klines(symbol='ETHUSDT', interval='1h', start_date='2025-01-01'):
    base_url = "https://api.binance.com/api/v3/klines"
    start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_time = int(datetime.now().timestamp() * 1000)
    limit = 1000

    all_data = []

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        response = requests.get(base_url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        start_time = data[-1][0] + 1
        time.sleep(0.5)

    return all_data


#  Загружаем данные
raw_data = get_binance_klines()

#  Преобразуем в DataFrame
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
           'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']
crypto = pd.DataFrame(raw_data, columns=columns)

#  Преобразование типов
crypto['timestamp'] = pd.to_datetime(crypto['timestamp'], unit='ms')
crypto = crypto[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
crypto[['open', 'high', 'low', 'close', 'volume']] = crypto[['open', 'high', 'low', 'close', 'volume']].astype(float)

#  Приводим время к часовому поясу Asia/Singapore
crypto.set_index('timestamp', inplace=True)
crypto.index = crypto.index.tz_localize('UTC').tz_convert('Asia/Singapore')

#  Переименовываем столбцы
crypto.reset_index(inplace=True)
crypto.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

#  Добавляем `time_num` для визуализации графиков
crypto['time_num'] = mdates.date2num(crypto['time'])

#  Анализ свечных паттернов с TA-Lib
candle_names = talib.get_function_groups()['Pattern Recognition']

# Убираем ненужные паттерны
removed = ['CDLCOUNTERATTACK', 'CDLLONGLINE', 'CDLSHORTLINE', 'CDLSTALLEDPATTERN', 'CDLKICKINGBYLENGTH']
candle_names = [name for name in candle_names if name not in removed]

#  Расчет свечных паттернов
op, hi, lo, cl = crypto['open'], crypto['high'], crypto['low'], crypto['close']

for candle in candle_names:
    crypto[candle] = getattr(talib, candle)(op, hi, lo, cl)

crypto.fillna(0, inplace=True)

#  Создаем столбцы для хранения паттернов
crypto['candlestick_pattern'] = ""
crypto['candlestick_match_count'] = 0

#  Присваиваем названия найденных паттернов
for index, row in crypto.iterrows():
    detected_patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))

    if len(detected_patterns) > 0:
        best_pattern = detected_patterns[0]  # Берем первый найденный паттерн
        label = "_Bull" if row[best_pattern] > 0 else "_Bear"
        crypto.at[index, 'candlestick_pattern'] = best_pattern + label
        crypto.at[index, 'candlestick_match_count'] = len(detected_patterns)


#  Функция нахождения уровней по экстремумам (локальные минимумы и максимумы)
def find_extreme_levels(data, order=100, min_touches=5):
    """
    Определяем уровни поддержки и сопротивления по локальным экстремумам.
    - order: число свечей по обе стороны для подтверждения экстремума
    - min_touches: минимальное количество касаний уровня, чтобы он считался значимым
    """
    prices = data['close'].values

    # Локальные максимумы (сопротивление)
    max_idx = argrelextrema(prices, np.greater, order=order)[0]

    # Локальные минимумы (поддержка)
    min_idx = argrelextrema(prices, np.less, order=order)[0]

    resistance_levels = data.iloc[max_idx]['close'].values
    support_levels = data.iloc[min_idx]['close'].values

    #  Фильтрация уровней: оставляем только те, которые тестировались min_touches раз
    support_levels = [level for level in support_levels if
                      (data['close'] - level).abs().lt(level * 0.01).sum() >= min_touches]
    resistance_levels = [level for level in resistance_levels if
                         (data['close'] - level).abs().lt(level * 0.01).sum() >= min_touches]

    return sorted(set(support_levels)), sorted(set(resistance_levels))

#  Находим уровни поддержки и сопротивления
support_levels, resistance_levels = find_extreme_levels(crypto)
print("Уровни поддержки:", support_levels)
print("Уровни сопротивления:", resistance_levels)


#  Функция проверки, находится ли паттерн около уровня экстремума
def is_near_extreme_level(price, levels, threshold=0.02):
    return any(abs(price - level) / price < threshold for level in levels)


#  Фильтрация паттернов, которые находятся около уровней экстремумов
crypto['near_extreme_level'] = crypto['close'].apply(
    lambda x: is_near_extreme_level(x, support_levels + resistance_levels))

patterns_near_extreme = crypto[
    (crypto['candlestick_pattern'] != "") &
    (crypto['candlestick_pattern'].notna()) &
    crypto['near_extreme_level']
    ]

#  Визуализация
output_folder = 'output/charts_extremes'
os.makedirs(output_folder, exist_ok=True)

for candle in candle_names:
    pattern_data = patterns_near_extreme[patterns_near_extreme['candlestick_pattern'].str.contains(candle)]

    if not pattern_data.empty:
        fig, ax = plt.subplots(figsize=(15, 7))

        ohlc = crypto[['time_num', 'open', 'high', 'low', 'close']].values
        candlestick_ohlc(ax, ohlc, width=0.0008, colorup='g', colordown='r')

        for level in support_levels :
            ax.axhline(level, linestyle='--', color='green', alpha=0.6)
            ax.text(crypto['time_num'].iloc[-1], level, f"{level:.2f}",
                    verticalalignment='center', fontsize=10, color='blue')

        for level in resistance_levels:
            ax.axhline(level, linestyle='--', color='red', alpha=0.6)
            ax.text(crypto['time_num'].iloc[-1], level, f"{level:.2f}",
                    verticalalignment='center', fontsize=10, color='blue')

        for index, row in pattern_data.iterrows():
            x = row['time_num']
            y_min, y_max = crypto['low'].min(), crypto['high'].max()

            if "_Bull" in row['candlestick_pattern']:
                ax.axvline(x, color='green', linestyle='--', alpha=0.7)
                ax.text(x, y_max * 1.05, "Bull", color='green', fontsize=10, rotation=90, verticalalignment='bottom')
            else:
                ax.axvline(x, color='red', linestyle='--', alpha=0.7)
                ax.text(x, y_min * 0.95, "Bear", color='red', fontsize=10, rotation=90, verticalalignment='top')

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.title(f"ETH/USDT - {candle} (только подтвержденные сигналы)")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True)

        plt.savefig(f"{output_folder}/{candle}_confirmed.png")
        plt.close()

print(f"Готово! Все графики сохранены в {output_folder}")
