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

#  Создаем столбцы для хранения паттернов (если не созданы)
if 'candlestick_pattern' not in crypto.columns:
    crypto['candlestick_pattern'] = ""
    crypto['candlestick_match_count'] = 0

#  Присваиваем названия найденных паттернов
for index, row in crypto.iterrows():
    detected_patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))

    if len(detected_patterns) > 0:
        pattern_name = detected_patterns[0]  # Берем первый найденный паттерн
        label = "_Bull" if row[pattern_name] > 0 else "_Bear"
        crypto.at[index, 'candlestick_pattern'] = pattern_name + label
        crypto.at[index, 'candlestick_match_count'] = len(detected_patterns)


#  Функция нахождения уровней по объемам (Volume Profile)
def find_volume_levels(data, bins=30, top_n_levels=10):
    volume_profile, bin_edges = np.histogram(data['close'], bins=bins, weights=data['volume'])
    sorted_indices = np.argsort(volume_profile)[-top_n_levels:]
    levels = bin_edges[sorted_indices]
    volumes = volume_profile[sorted_indices]
    return sorted(levels), volumes


#  Функция проверки, находится ли паттерн около объёмного уровня
def is_near_volume_level(price, levels, threshold=0.02):
    return any(abs(price - level) / price < threshold for level in levels)


#  Находим уровни по объемам
volume_levels, volume_values = find_volume_levels(crypto, bins=30, top_n_levels=10)

#  Проверяем, пуст ли список уровней
if len(volume_levels) > 0:
    crypto['near_volume_level'] = crypto['close'].apply(lambda x: is_near_volume_level(x, volume_levels))
    patterns_near_volume = crypto[
        (crypto['candlestick_pattern'] != "") &
        (crypto['candlestick_pattern'].notna()) &
        (crypto['near_volume_level'])
        ]
else:
    print("Ошибка: не удалось определить уровни объема!")
    patterns_near_volume = pd.DataFrame()


output_folder = 'output/charts-volume-levels2'
os.makedirs(output_folder, exist_ok=True)  # Создаем папку, если её нет

#  Визуализация графиков с объемными уровнями и подтвержденными паттернами
for candle in candle_names:
    pattern_data = patterns_near_volume[patterns_near_volume['candlestick_pattern'].str.contains(candle, na=False)]

    if not pattern_data.empty:
        fig, ax = plt.subplots(figsize=(15, 7))

        #  Форматируем данные для `candlestick_ohlc`
        ohlc = crypto[['time_num', 'open', 'high', 'low', 'close']].values
        candlestick_ohlc(ax, ohlc, width=0.0008, colorup='g', colordown='r')

        #  Добавляем объемные уровни
        for level, volume in zip(volume_levels, volume_values):
            ax.axhline(level, linestyle='--', color='blue', alpha=0.6)
            ax.text(crypto['time_num'].iloc[-1], level, f"{level:.2f} ({int(volume)})",
                    verticalalignment='center', fontsize=10, color='blue')

        #  Добавляем вертикальные линии для паттернов
        for index, row in pattern_data.iterrows():
            x = row['time_num']
            y_max = crypto['high'].max()
            y_min = crypto['low'].min()

            if "_Bull" in row['candlestick_pattern']:
                ax.axvline(x, color='green', linestyle='--', alpha=0.7)
                ax.text(x, y_max * 1.05, row['candlestick_pattern'],
                        color='green', fontsize=10, rotation=90, verticalalignment='bottom')
            else:
                ax.axvline(x, color='red', linestyle='--', alpha=0.7)
                ax.text(x, y_min * 0.95, row['candlestick_pattern'],
                        color='red', fontsize=10, rotation=90, verticalalignment='top')

        #  Настройки графика
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.title(f"ETH/USDT - {candle} (только подтвержденные сигналы)")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True)

        #  Сохранение графика
        plt.savefig(f"{output_folder}/{candle}_confirmed.png")
        plt.close()

print(f"Готово! Все графики сохранены в {output_folder}")
