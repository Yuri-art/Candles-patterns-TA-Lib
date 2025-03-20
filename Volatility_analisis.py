import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class ETHVolatilityAnalyzer:
    def __init__(self, api_base_url='https://api.binance.com'):
        """
        Инициализирует анализатор волатильности ETH

        Parameters:
        api_base_url (str): Базовый URL API Binance
        """
        self.api_base_url = api_base_url
        self.df = None

    def fetch_historical_data(self, symbol='ETHUSDT', interval='1h',
                              analysis_period_days=365, sma_window_days=14):
        """
        Загружает исторические данные с Binance API

        Parameters:
        symbol (str): Торговая пара
        interval (str): Интервал свечей ('1h', '4h', '1d' и т.д.)
        analysis_period_days (int): Основной период анализа в днях
        sma_window_days (int): Размер окна для скользящей средней в днях

        Returns:
        pandas.DataFrame: Исторические данные с OHLCV
        """
        endpoint = '/api/v3/klines'

        # Рассчитываем, сколько данных нам нужно загрузить
        total_period_days = analysis_period_days + sma_window_days

        # Рассчитываем количество интервалов в зависимости от выбранного периода
        intervals_per_day = 24  # для часовых свечей
        if interval == '4h':
            intervals_per_day = 6
        elif interval == '1d':
            intervals_per_day = 1

        # Общее количество интервалов, которые необходимо получить
        limit = total_period_days * intervals_per_day

        # Binance API ограничивает выдачу 1000 записей за запрос,
        # поэтому делим запрос на части, если нужно
        max_limit_per_request = 1000
        all_klines = []

        end_time = int(time.time() * 1000)  # текущее время в миллисекундах

        while limit > 0:
            request_limit = min(limit, max_limit_per_request)
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': request_limit,
                'endTime': end_time
            }

            response = requests.get(f"{self.api_base_url}{endpoint}", params=params)
            klines = response.json()

            if not klines:
                break

            all_klines = klines + all_klines
            limit -= len(klines)

            # Обновляем end_time для следующего запроса
            end_time = klines[0][0] - 1  # Время открытия первой свечи минус 1 мс

        # Преобразуем данные в DataFrame
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                   'Close time', 'Quote asset volume', 'Number of trades',
                   'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

        df = pd.DataFrame(all_klines, columns=columns)

        # Преобразуем типы данных
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])

        # Преобразуем временные метки в datetime
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

        # Сортируем по времени
        df = df.sort_values('Open time')

        # Сохраняем DataFrame
        self.df = df

        print(f"Загружено {len(df)} записей из Binance API.")
        return df

    def calculate_volatility(self, analysis_period_days=365, sma_window_days=14,
                             thresholds=[3, 5, 7, 10]):
        """
        Рассчитывает волатильность ETH относительно скользящей средней

        Parameters:
        analysis_period_days (int): Основной период анализа в днях
        sma_window_days (int): Размер окна для скользящей средней в днях
        thresholds (list): Пороговые значения отклонений в процентах

        Returns:
        dict: Результаты анализа
        """
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала вызовите fetch_historical_data().")

        # Создаем копию DataFrame для анализа
        df = self.df.copy()

        # Рассчитываем скользящую среднюю
        hours_in_window = sma_window_days * 24
        df['SMA'] = df['Close'].rolling(window=hours_in_window).mean()

        # Отбрасываем записи, где SMA не определена (первые sma_window_days)
        df = df.dropna(subset=['SMA'])

        # Берем только данные за последний год (analysis_period_days)
        hours_in_analysis = analysis_period_days * 24
        if len(df) > hours_in_analysis:
            df = df.iloc[-hours_in_analysis:]

        # Рассчитываем максимальные отклонения для High и Low
        df['High_deviation'] = abs((df['High'] - df['SMA']) / df['SMA']) * 100
        df['Low_deviation'] = abs((df['Low'] - df['SMA']) / df['SMA']) * 100

        # Находим максимальное отклонение для каждого интервала
        df['Max_deviation'] = df[['High_deviation', 'Low_deviation']].max(axis=1)

        # Считаем количество превышений каждого порога
        results = {}
        total_intervals = len(df)

        for threshold in thresholds:
            count = len(df[df['Max_deviation'] > threshold])
            percentage = (count / total_intervals) * 100
            results[f"Отклонение > {threshold}%"] = {
                'Количество интервалов': count,
                'Процент от общего': f"{percentage:.2f}%"
            }

        # Добавляем основную информацию
        results['Общая информация'] = {
            'Всего интервалов': total_intervals,
            'Период анализа (дней)': analysis_period_days,
            'Окно скользящей средней (дней)': sma_window_days,
            'Начальная дата': df['Open time'].min().strftime('%Y-%m-%d'),
            'Конечная дата': df['Open time'].max().strftime('%Y-%m-%d')
        }

        # Сохраняем результаты анализа
        self.results = results
        self.analysis_df = df

        return results

    def plot_results(self, save_path=None):
        """
        Визуализирует результаты анализа

        Parameters:
        save_path (str, optional): Путь для сохранения графика
        """
        if not hasattr(self, 'analysis_df') or not hasattr(self, 'results'):
            raise ValueError("Анализ не выполнен. Сначала вызовите calculate_volatility().")

        df = self.analysis_df

        # Создаем фигуру с подграфиками
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # График 1: Цена ETH и скользящая средняя
        ax1.plot(df['Open time'], df['Close'], label='ETH Close Price', alpha=0.7)
        ax1.plot(df['Open time'], df['SMA'],
                 label=f"{self.results['Общая информация']['Окно скользящей средней (дней)']} Day SMA",
                 color='red', linewidth=2)
        ax1.set_title('ETH Price and SMA')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Отклонения от скользящей средней
        ax2.plot(df['Open time'], df['Max_deviation'], label='Max Deviation (%)', color='purple', alpha=0.7)

        # Добавляем горизонтальные линии для пороговых значений
        thresholds = [float(key.split('%')[0].split('> ')[1]) for key in self.results.keys()
                      if key.startswith('Отклонение')]

        for threshold in thresholds:
            ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.5,
                        label=f"{threshold}% Threshold")

        ax2.set_title('Maximum Deviation from SMA (%)')
        ax2.set_ylabel('Deviation (%)')
        ax2.set_ylim(bottom=0)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Общие настройки
        plt.tight_layout()

        # Сохраняем график, если указан путь
        if save_path:
            plt.savefig(save_path)

        return fig

    def print_results(self):
        """
        Выводит результаты анализа в удобочитаемом формате
        """
        if not hasattr(self, 'results'):
            raise ValueError("Анализ не выполнен. Сначала вызовите calculate_volatility().")

        print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ВОЛАТИЛЬНОСТИ ETH ===\n")

        # Выводим общую информацию
        print("Общая информация:")
        for key, value in self.results['Общая информация'].items():
            print(f"- {key}: {value}")

        print("\nОтклонения от скользящей средней:")
        for key, value in self.results.items():
            if key != 'Общая информация':
                print(f"\n{key}:")
                for stat_key, stat_value in value.items():
                    print(f"- {stat_key}: {stat_value}")


# Пример использования
if __name__ == "__main__":
    # Параметры анализа
    symbol = 'CAKEUSDT'
    interval = '1h'
    analysis_period_days = 365
    sma_window_days = 1
    thresholds = [3, 5, 10, 15, 20]

    # Создаем анализатор
    analyzer = ETHVolatilityAnalyzer()

    # Загружаем данные
    analyzer.fetch_historical_data(
        symbol=symbol,
        interval=interval,
        analysis_period_days=analysis_period_days,
        sma_window_days=sma_window_days
    )

    # Рассчитываем отклонения
    results = analyzer.calculate_volatility(
        analysis_period_days=analysis_period_days,
        sma_window_days=sma_window_days,
        thresholds=thresholds
    )

    # Выводим результаты
    analyzer.print_results()

    # Визуализируем результаты
    fig = analyzer.plot_results(save_path="eth_volatility_analysis.png")
    plt.show()