import warnings
import itertools
import time
import os
import pandas as pd
import numpy as np
import talib
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from datetime import datetime
from itertools import compress
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore", category=FutureWarning)

# 📌 ФИКСИРОВАННЫЕ ПАРАМЕТРЫ
# Минимальная требуемая точность предсказаний для сохранения графика
MIN_ACCURACY_THRESHOLD = 60.0  # % минимальная точность
# Минимальное количество предсказаний для учета в оценке точности
MIN_PREDICTIONS_COUNT = 3  # Должно быть не менее 3 предсказаний

# 📌 ПАРАМЕТРЫ ДЛЯ ОПТИМИЗАЦИИ
# Параметры определения уровней поддержки и сопротивления - варианты для перебора
EXTREMA_ORDER_VALUES = [50, 100, 150]  # Количество свечей для определения локальных экстремумов
MIN_LEVEL_TOUCHES_VALUES = [3, 5, 7]  # Минимальное количество касаний уровня для его подтверждения
LEVEL_TOUCH_THRESHOLD_VALUES = [0.5, 1.0, 1.5]  # Процент отклонения от уровня, считающийся касанием (в %)

# Параметр близости паттерна к уровню - варианты для перебора
LEVEL_PROXIMITY_THRESHOLD_VALUES = [2.0, 3.0]  # Расстояние от уровня поддержки/сопротивления (%)

# Параметры для бычьих и медвежьих паттернов - варианты для перебора
TARGET_PERCENT_VALUES = [3.0, 4.0]  # Целевой рост/падение (%)
STOP_PERCENT_VALUES = [1.0, 1.5]  # Максимально допустимое противоположное движение (%)


# 📌 Функция для загрузки данных ETH/USDT с Binance
def get_binance_klines(symbol='ETHUSDT', interval='1h', start_date='2024-08-01'):
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


# 📌 Функция нахождения уровней по экстремумам
def find_extreme_levels(data, order, min_touches, touch_threshold):
    prices = data['close'].values
    max_idx = argrelextrema(prices, np.greater, order=order)[0]
    min_idx = argrelextrema(prices, np.less, order=order)[0]

    resistance_levels = [(data.iloc[i]['close'], data.iloc[i]['time']) for i in max_idx]
    support_levels = [(data.iloc[i]['close'], data.iloc[i]['time']) for i in min_idx]

    # Фильтруем уровни по количеству касаний
    support_levels = [(level, time) for level, time in support_levels
                      if (data['close'] - level).abs().lt(level * touch_threshold / 100).sum() >= min_touches]
    resistance_levels = [(level, time) for level, time in resistance_levels
                         if (data['close'] - level).abs().lt(level * touch_threshold / 100).sum() >= min_touches]

    return support_levels, resistance_levels


# 📌 Функция проверки паттерна относительно уровней
def check_pattern_levels(price, time, support_levels, resistance_levels, level_proximity_threshold):
    # Словарь для хранения результатов проверки
    results = {
        'standard_bull': False,  # Бычий у поддержки
        'standard_bear': False,  # Медвежий у сопротивления
        'logical_bull': False,  # Бычий у сопротивления
        'logical_bear': False  # Медвежий у поддержки
    }

    # Проверка у уровней поддержки (все паттерны должны быть после формирования уровня)
    for level, level_time in support_levels:
        if abs(price - level) / price < level_proximity_threshold / 100 and time > level_time:
            results['standard_bull'] = True  # Для бычьих паттернов у поддержки
            results['logical_bear'] = True  # Для медвежьих паттернов у поддержки
            break

    # Проверка у уровней сопротивления (все паттерны должны быть после формирования уровня)
    for level, level_time in resistance_levels:
        if abs(price - level) / price < level_proximity_threshold / 100 and time > level_time:
            results['standard_bear'] = True  # Для медвежьих паттернов у сопротивления
            results['logical_bull'] = True  # Для бычьих паттернов у сопротивления
            break

    return results


# 📌 Функция для анализа эффективности паттернов
def analyze_pattern_performance(data, pattern_row, pattern_type, target_percent, stop_percent):
    """
    Анализирует эффективность паттерна на основе последующего движения цены

    Args:
        data (pd.DataFrame): DataFrame с данными
        pattern_row (pd.Series): Строка с данными о паттерне
        pattern_type (str): Тип паттерна (Bull, L_Bull, Bear, L_Bear)
        target_percent (float): Целевой процент роста/падения
        stop_percent (float): Максимально допустимое противоположное движение

    Returns:
        bool: True если паттерн был успешным, False в противном случае
    """
    # Индекс строки с паттерном
    pattern_idx = pattern_row.name

    # Цена на момент появления паттерна
    pattern_price = pattern_row['close']

    # Находим данные после паттерна (до конца DataFrame)
    future_data = data.loc[pattern_idx + 1:]

    # Если нет данных после паттерна, считаем его неуспешным
    if future_data.empty:
        return False

    # Для бычьих паттернов (Bull и L_Bull)
    if "Bull" in pattern_type:
        # Проверяем, был ли рост на целевой процент и не было ли падения более чем на допустимый процент
        min_price_threshold = pattern_price * (1 - stop_percent / 100)  # -X%
        target_price = pattern_price * (1 + target_percent / 100)  # +Y%

        # Для каждой свечи после паттерна проверяем условия
        for idx, row in future_data.iterrows():
            # Если цена упала более чем на допустимый процент - условие не выполнено
            if row['low'] < min_price_threshold:
                return False

            # Если цена выросла как минимум на целевой процент - условие выполнено
            if row['high'] >= target_price:
                return True

    # Для медвежьих паттернов (Bear и L_Bear)
    elif "Bear" in pattern_type:
        # Проверяем, было ли падение на целевой процент и не было ли роста более чем на допустимый процент
        max_price_threshold = pattern_price * (1 + stop_percent / 100)  # +X%
        target_price = pattern_price * (1 - target_percent / 100)  # -Y%

        # Для каждой свечи после паттерна проверяем условия
        for idx, row in future_data.iterrows():
            # Если цена выросла более чем на допустимый процент - условие не выполнено
            if row['high'] > max_price_threshold:
                return False

            # Если цена упала как минимум на целевой процент - условие выполнено
            if row['low'] <= target_price:
                return True

    # Если мы дошли до конца данных, но целевая цена не была достигнута
    return False


# 📌 Функция для вычисления средней точности по всем присутствующим паттернам
def calculate_average_accuracy(pattern_stats):
    total_success = 0
    total_patterns = 0

    for pattern_type, stats in pattern_stats.items():
        if stats['total'] > 0:
            total_success += stats['success']
            total_patterns += stats['total']

    # Проверяем, что общее количество паттернов достаточно для надежной оценки
    if total_patterns >= MIN_PREDICTIONS_COUNT:
        return (total_success / total_patterns) * 100, total_patterns
    else:
        return 0, total_patterns  # Недостаточно данных для надежной оценки


# 📌 Функция для обработки данных и визуализации с заданными параметрами
def process_crypto_data(data, extrema_order, min_touches, touch_threshold,
                        level_proximity_threshold, target_percent, stop_percent):
    # Создаем копию данных для работы
    crypto = data.copy()

    # 📌 Находим уровни поддержки и сопротивления с текущими параметрами
    support_levels, resistance_levels = find_extreme_levels(
        crypto, extrema_order, min_touches, touch_threshold
    )

    # 📌 Очищаем колонки для хранения информации о паттернах
    crypto['standard_pattern'] = False
    crypto['logical_pattern'] = False
    crypto['pattern_type'] = ""

    # 📌 Применяем проверку к каждой строке
    for index, row in crypto.iterrows():
        if row['candlestick_pattern']:
            check_results = check_pattern_levels(
                row['close'],
                row['time'],
                support_levels,
                resistance_levels,
                level_proximity_threshold
            )

            is_bull = "_Bull" in row['candlestick_pattern']
            is_bear = "_Bear" in row['candlestick_pattern']

            # Проверяем стандартные паттерны
            if (is_bull and check_results['standard_bull']) or (is_bear and check_results['standard_bear']):
                crypto.at[index, 'standard_pattern'] = True
                crypto.at[index, 'pattern_type'] = "standard"

            # Проверяем логические паттерны
            elif (is_bull and check_results['logical_bull']):
                crypto.at[index, 'logical_pattern'] = True
                crypto.at[index, 'pattern_type'] = "logical_bull"

            elif (is_bear and check_results['logical_bear']):
                crypto.at[index, 'logical_pattern'] = True
                crypto.at[index, 'pattern_type'] = "logical_bear"

    # 📌 Фильтруем паттерны для визуализации
    all_filtered_patterns = crypto[(crypto['standard_pattern']) | (crypto['logical_pattern'])]

    # 📌 Результаты для каждого типа паттерна
    results_by_candle = {}

    # 📌 Обрабатываем каждый тип свечного паттерна
    for candle in candle_names:
        # Выбираем все паттерны данного типа
        pattern_data = all_filtered_patterns[all_filtered_patterns['candlestick_pattern'].str.contains(candle)]

        if not pattern_data.empty:
            # Создаем словарь для хранения статистики по типам паттернов
            pattern_stats = {
                'Bull': {'total': 0, 'success': 0},
                'Bear': {'total': 0, 'success': 0},
                'L_Bull': {'total': 0, 'success': 0},
                'L_Bear': {'total': 0, 'success': 0}
            }

            # Анализ эффективности каждого паттерна
            for index, row in pattern_data.iterrows():
                pattern_type = ""

                # Определяем тип паттерна
                if row['pattern_type'] == "standard":
                    if "_Bull" in row['candlestick_pattern']:
                        pattern_type = "Bull"
                    else:
                        pattern_type = "Bear"
                elif row['pattern_type'] == "logical_bull":
                    pattern_type = "L_Bull"
                elif row['pattern_type'] == "logical_bear":
                    pattern_type = "L_Bear"

                # Увеличиваем счетчик общего количества паттернов
                pattern_stats[pattern_type]['total'] += 1

                # Проверяем эффективность паттерна
                if analyze_pattern_performance(crypto, row, pattern_type, target_percent, stop_percent):
                    pattern_stats[pattern_type]['success'] += 1

            # Рассчитываем среднюю точность по всем паттернам на графике
            avg_accuracy, total_predictions = calculate_average_accuracy(pattern_stats)

            # Сохраняем результаты для этого типа свечного паттерна
            results_by_candle[candle] = {
                'pattern_stats': pattern_stats,
                'avg_accuracy': avg_accuracy,
                'total_predictions': total_predictions,
                'pattern_data': pattern_data
            }

    return {
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'results_by_candle': results_by_candle
    }


# 📌 Функция для создания и сохранения графика
def create_and_save_chart(crypto, candle, pattern_data, pattern_stats, avg_accuracy,
                          support_levels, resistance_levels, params, output_folder, total_predictions):
    # Создаем график только если средняя точность выше порогового значения
    # и количество предсказаний достаточно для надежной оценки
    if avg_accuracy >= MIN_ACCURACY_THRESHOLD and total_predictions >= MIN_PREDICTIONS_COUNT:
        try:
            fig, ax = plt.subplots(figsize=(15, 9))

            ohlc = crypto[['time_num', 'open', 'high', 'low', 'close']].values
            candlestick_ohlc(ax, ohlc, width=0.0008, colorup='g', colordown='r')

            # 📌 Добавляем уровни поддержки (зеленые) и сопротивления (красные)
            for level, level_time in support_levels:
                ax.axhline(level, linestyle='--', color='green', alpha=0.6)
                ax.text(crypto['time_num'].iloc[-1], level, f"{level:.2f} ({level_time})", fontsize=10, color='blue')

            for level, level_time in resistance_levels:
                ax.axhline(level, linestyle='--', color='red', alpha=0.6)
                ax.text(crypto['time_num'].iloc[-1], level, f"{level:.2f} ({level_time})", fontsize=10, color='blue')

            # 📌 Добавляем вертикальные линии для паттернов
            for index, row in pattern_data.iterrows():
                x = row['time_num']
                y_max = crypto['high'].max()
                y_min = crypto['low'].min()

                # Стандартные паттерны - как в оригинальном коде
                if row['pattern_type'] == "standard":
                    if "_Bull" in row['candlestick_pattern']:
                        ax.axvline(x, color='green', linestyle='--', alpha=0.7)
                        ax.text(x, y_max * 1.05, "Bull", color='green', fontsize=10, rotation=90,
                                verticalalignment='bottom')
                    else:
                        ax.axvline(x, color='red', linestyle='--', alpha=0.7)
                        ax.text(x, y_min * 0.95, "Bear", color='red', fontsize=10, rotation=90, verticalalignment='top')

                # Логические бычьи паттерны у сопротивления (синие)
                elif row['pattern_type'] == "logical_bull":
                    ax.axvline(x, color='blue', linestyle='--', alpha=0.7)
                    ax.text(x, y_max * 1.05, "L_Bull", color='blue', fontsize=10, rotation=90,
                            verticalalignment='bottom')

                # Логические медвежьи паттерны у поддержки (черные)
                elif row['pattern_type'] == "logical_bear":
                    ax.axvline(x, color='black', linestyle='--', alpha=0.7)
                    ax.text(x, y_min * 0.95, "L_Bear", color='black', fontsize=10, rotation=90, verticalalignment='top')

            # 📌 Настройки графика
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)

            # Добавляем информацию о параметрах в заголовок
            plt.title(f"ETH/USDT - {candle} | Точность: {avg_accuracy:.1f}% | Прогнозов: {total_predictions} | " +
                      f"Ext:{params['extrema_order']} Touch:{params['min_touches']} " +
                      f"Prox:{params['level_proximity_threshold']}% Tgt:{params['target_percent']}%")

            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.grid(True)

            # 📌 Добавляем блок со статистикой внизу графика
            stats_text = f"Параметры: Экстремумы {params['extrema_order']} свечей, Мин. касаний {params['min_touches']}, " + \
                         f"Порог касания {params['touch_threshold']}%, Близость к уровню {params['level_proximity_threshold']}%, " + \
                         f"Цель {params['target_percent']}%, Стоп {params['stop_percent']}%\n\n"

            stats_text += f"Средняя точность предсказаний: {avg_accuracy:.1f}% (на основе {total_predictions} прогнозов)\n"
            stats_text += f"Мин. требуемая точность: {MIN_ACCURACY_THRESHOLD}%, Мин. количество прогнозов: {MIN_PREDICTIONS_COUNT}\n"
            stats_text += "Статистика эффективности паттернов:\n"

            # Рассчитываем процент успешности для каждого типа паттерна
            for pattern_type, stats in pattern_stats.items():
                if stats['total'] > 0:
                    success_rate = (stats['success'] / stats['total']) * 100
                    stats_text += f"{pattern_type}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)\n"
                else:
                    stats_text += f"{pattern_type}: 0/0 (0.0%)\n"

            # Добавляем текстовое поле со статистикой под графиком
            plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=12,
                        bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5'))

            plt.tight_layout(rect=[0, 0.13, 1, 1])  # Увеличиваем отступ снизу для текста статистики

            # Создаем имя файла, включающее значения параметров
            # Безопасное имя файла без запрещенных символов
            safe_candle_name = candle.replace("/", "_").replace("\\", "_")
            filename = f"{safe_candle_name}_acc{avg_accuracy:.1f}_pred{total_predictions}_ext{params['extrema_order']}" + \
                       f"_touch{params['min_touches']}_th{params['touch_threshold']}" + \
                       f"_prox{params['level_proximity_threshold']}_tgt{params['target_percent']}_stop{params['stop_percent']}.png"

            # 📌 Сохранение графика
            plt.savefig(f"{output_folder}/{filename}")
            plt.close()

            # Проверяем, действительно ли файл был создан
            if os.path.exists(f"{output_folder}/{filename}"):
                print(f"✅ Сохранен график: {filename}")
                return True  # График был успешно создан и сохранен
            else:
                print(f"❌ Ошибка при сохранении графика: {filename}")
                return False

        except Exception as e:
            print(f"❌ Ошибка при создании графика {candle}: {str(e)}")
            return False

    return False  # График не был создан из-за низкой точности или недостаточного количества предсказаний


# 📌 Основная функция оптимизации параметров
def optimize_parameters():
    print("Начинаем загрузку данных и подготовку к оптимизации...")

    # 📌 Загружаем данные
    raw_data = get_binance_klines()

    # 📌 Преобразуем в DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
               'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore']
    crypto = pd.DataFrame(raw_data, columns=columns)

    # 📌 Преобразование типов
    crypto['timestamp'] = pd.to_datetime(crypto['timestamp'], unit='ms')
    crypto = crypto[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    crypto[['open', 'high', 'low', 'close', 'volume']] = crypto[['open', 'high', 'low', 'close', 'volume']].astype(
        float)

    # 📌 Приводим время к часовому поясу Asia/Singapore
    crypto.set_index('timestamp', inplace=True)
    crypto.index = crypto.index.tz_localize('UTC').tz_convert('Asia/Singapore')

    # 📌 Переименовываем столбцы
    crypto.reset_index(inplace=True)
    crypto.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

    # 📌 Добавляем `time_num` для визуализации графиков
    crypto['time_num'] = mdates.date2num(crypto['time'])

    # 📌 Анализ свечных паттернов с TA-Lib
    global candle_names
    candle_names = talib.get_function_groups()['Pattern Recognition']

    # Убираем ненужные паттерны
    removed = ['CDLCOUNTERATTACK', 'CDLLONGLINE', 'CDLSHORTLINE', 'CDLSTALLEDPATTERN', 'CDLKICKINGBYLENGTH']
    candle_names = [name for name in candle_names if name not in removed]

    # 📌 Расчет свечных паттернов
    op, hi, lo, cl = crypto['open'], crypto['high'], crypto['low'], crypto['close']

    for candle in candle_names:
        crypto[candle] = getattr(talib, candle)(op, hi, lo, cl)

    crypto.fillna(0, inplace=True)

    # 📌 Создаем столбцы для хранения паттернов
    crypto['candlestick_pattern'] = ""
    crypto['candlestick_match_count'] = 0

    # 📌 Присваиваем названия найденных паттернов
    for index, row in crypto.iterrows():
        detected_patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))

        if len(detected_patterns) > 0:
            best_pattern = detected_patterns[0]  # Берем первый найденный паттерн
            label = "_Bull" if row[best_pattern] > 0 else "_Bear"
            crypto.at[index, 'candlestick_pattern'] = best_pattern + label
            crypto.at[index, 'candlestick_match_count'] = len(detected_patterns)

    # 📌 Создаем папку для сохранения результатов
    output_folder = 'output/optimized_charts'
    os.makedirs(output_folder, exist_ok=True)

    # 📌 Счетчики для статистики
    total_combinations = (len(EXTREMA_ORDER_VALUES) * len(MIN_LEVEL_TOUCHES_VALUES) *
                          len(LEVEL_TOUCH_THRESHOLD_VALUES) * len(LEVEL_PROXIMITY_THRESHOLD_VALUES) *
                          len(TARGET_PERCENT_VALUES) * len(STOP_PERCENT_VALUES))

    saved_charts = 0
    best_accuracy = 0
    best_params = {}

    print(f"Всего комбинаций параметров для перебора: {total_combinations}")
    start_time = time.time()

    # 📌 Создаем CSV-файл для записи результатов
    results_file = f"{output_folder}/optimization_results.csv"
    with open(results_file, 'w') as f:
        f.write("extrema_order,min_touches,touch_threshold,level_proximity_threshold," +
                "target_percent,stop_percent,candle_name,avg_accuracy,total_predictions\n")

    # 📌 Перебираем все комбинации параметров
    combination_counter = 0
    for extrema_order in EXTREMA_ORDER_VALUES:
        for min_touches in MIN_LEVEL_TOUCHES_VALUES:
            for touch_threshold in LEVEL_TOUCH_THRESHOLD_VALUES:
                for level_proximity_threshold in LEVEL_PROXIMITY_THRESHOLD_VALUES:
                    for target_percent in TARGET_PERCENT_VALUES:
                        for stop_percent in STOP_PERCENT_VALUES:
                            combination_counter += 1

                            # Выводим прогресс
                            if combination_counter % 10 == 0 or combination_counter == 1:
                                elapsed_time = time.time() - start_time
                                eta = (elapsed_time / combination_counter) * (total_combinations - combination_counter)
                                print(f"Обработано {combination_counter}/{total_combinations} комбинаций " +
                                      f"({combination_counter / total_combinations * 100:.1f}%) | " +
                                      f"Прошло: {elapsed_time / 60:.1f} мин | " +
                                      f"Осталось: {eta / 60:.1f} мин | " +
                                      f"Сохранено графиков: {saved_charts}")

                            # Текущие параметры
                            params = {
                                'extrema_order': extrema_order,
                                'min_touches': min_touches,
                                'touch_threshold': touch_threshold,
                                'level_proximity_threshold': level_proximity_threshold,
                                'target_percent': target_percent,
                                'stop_percent': stop_percent
                            }

                            # Обрабатываем данные с текущими параметрами
                            results = process_crypto_data(
                                crypto,
                                extrema_order,
                                min_touches,
                                touch_threshold,
                                level_proximity_threshold,
                                target_percent,
                                stop_percent
                            )

                            # Для каждого свечного паттерна проверяем результаты и сохраняем график, если точность высокая
                            for candle, candle_results in results['results_by_candle'].items():
                                avg_accuracy = candle_results['avg_accuracy']
                                total_predictions = candle_results['total_predictions']

                                # Записываем результаты в CSV
                                with open(results_file, 'a') as f:
                                    f.write(f"{extrema_order},{min_touches},{touch_threshold}," +
                                            f"{level_proximity_threshold},{target_percent},{stop_percent}," +
                                            f"{candle},{avg_accuracy:.1f},{total_predictions}\n")

                                # Проверяем, достаточно ли высока точность для сохранения графика
                                # и достаточно ли предсказаний для надежной оценки
                                if avg_accuracy >= MIN_ACCURACY_THRESHOLD and total_predictions >= MIN_PREDICTIONS_COUNT:
                                    chart_saved = create_and_save_chart(
                                        crypto,
                                        candle,
                                        candle_results['pattern_data'],
                                        candle_results['pattern_stats'],
                                        avg_accuracy,
                                        results['support_levels'],
                                        results['resistance_levels'],
                                        params,
                                        output_folder,
                                        total_predictions
                                    )

                                    if chart_saved:
                                        saved_charts += 1
                                        print(f"Комбинация {combination_counter}/{total_combinations}: " +
                                              f"✅ сохранен график {saved_charts} для {candle} " +
                                              f"(точность: {avg_accuracy:.1f}%, прогнозов: {total_predictions}, " +
                                              f"параметры: ext={params['extrema_order']}, touch={params['min_touches']}, " +
                                              f"th={params['touch_threshold']}, prox={params['level_proximity_threshold']}, " +
                                              f"tgt={params['target_percent']}, stop={params['stop_percent']})")
                                    else:
                                        print(f"Комбинация {combination_counter}/{total_combinations}: " +
                                              f"❌ не удалось сохранить график для {candle} " +
                                              f"(точность: {avg_accuracy:.1f}%, прогнозов: {total_predictions})")

                                    # Обновляем информацию о лучших параметрах
                                    if avg_accuracy > best_accuracy:
                                        best_accuracy = avg_accuracy
                                        best_params = {
                                            'candle': candle,
                                            **params
                                        }

    # 📌 Выводим итоговую статистику
    total_time = time.time() - start_time
    print(f"\n✅ Оптимизация завершена!")
    print(f"Всего проверено комбинаций: {total_combinations}")
    print(
        f"Сохранено графиков с точностью >= {MIN_ACCURACY_THRESHOLD}% и количеством прогнозов >= {MIN_PREDICTIONS_COUNT}: {saved_charts}")
    print(f"Общее время выполнения: {total_time / 60:.1f} минут")

    if best_params:
        print(f"\nЛучшая найденная комбинация параметров:")
        print(f"Свечной паттерн: {best_params['candle']}")
        print(f"Период экстремумов: {best_params['extrema_order']}")
        print(f"Мин. касаний уровня: {best_params['min_touches']}")
        print(f"Порог касания: {best_params['touch_threshold']}%")
        print(f"Близость к уровню: {best_params['level_proximity_threshold']}%")
        print(f"Целевой процент: {best_params['target_percent']}%")
        print(f"Стоп-процент: {best_params['stop_percent']}%")
        print(f"Точность: {best_accuracy:.1f}%")

    # Создаем сводный CSV с топ-10 лучших комбинаций
    try:
        results_df = pd.read_csv(results_file)
        top_results = results_df.sort_values('avg_accuracy', ascending=False).head(10)
        top_file = f"{output_folder}/top10_combinations.csv"
        top_results.to_csv(top_file, index=False)
        print(f"\nТоп-10 лучших комбинаций сохранены в файл: {top_file}")
    except Exception as e:
        print(f"Ошибка при создании файла с топ-10 комбинаций: {e}")

    return best_params, best_accuracy


# Запускаем оптимизацию, если скрипт выполняется напрямую
if __name__ == "__main__":
    optimize_parameters()