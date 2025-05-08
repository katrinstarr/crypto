
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# Функции для расчёта индикаторов
# =============================================================================

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def calculate_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def calculate_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    up_move = high.diff()
    down_move = low.diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period, min_periods=period).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period, min_periods=period).sum() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).fillna(0)
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx

# =============================================================================
# Функции для работы с данными биржи и расчёта объёма позиции
# =============================================================================

def fetch_ohlcv(exchange, symbol, timeframe='1d', limit=100):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def get_top_pairs(exchange, limit=10):
    tickers = exchange.fetch_tickers()
    usdt_pairs = {symbol: ticker for symbol, ticker in tickers.items() if '/USDT' in symbol}
    sorted_pairs = sorted(usdt_pairs.items(), key=lambda x: x[1].get('quoteVolume', 0), reverse=True)
    return [symbol for symbol, _ in sorted_pairs[:limit]]

def calculate_position_size(deposit, risk_percent, entry, stop_loss):
    risk_amount = deposit * risk_percent
    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit == 0:
        return 0
    return risk_amount / risk_per_unit

# =============================================================================
# Основная функция анализа с улучшенной адаптивной стратегией
# =============================================================================

def analyze_symbol_improved(exchange, symbol):
    # Загружаем дневные данные за 70 свечей (70 > 20, что позволяет рассчитать наклон MA)
    df = fetch_ohlcv(exchange, symbol, timeframe='1d', limit=70)
    
    # Расчёт индикаторов
    df['rsi'] = calculate_rsi(df['close'])
    macd_line, signal_line, macd_hist = calculate_macd(df['close'])
    df['macd_hist'] = macd_hist
    df['ma20'] = df['close'].rolling(window=20, min_periods=20).mean()
    df['atr'] = calculate_atr(df, period=14)
    df['adx'] = calculate_adx(df, period=14)
    
    # Bollinger Bands (для fallback в диапазонном режиме)
    df['std20'] = df['close'].rolling(window=20, min_periods=20).std()
    df['lower_band'] = df['ma20'] - 2 * df['std20']
    df['upper_band'] = df['ma20'] + 2 * df['std20']
    
    # Выбираем последнюю свечу
    current = df.iloc[-1]
    current_price = current['close']
    current_rsi = current['rsi']
    current_macd = current['macd_hist']
    current_ma20 = current['ma20']
    current_atr = current['atr']
    current_adx = current['adx']
    lower_band = current['lower_band']
    upper_band = current['upper_band']
    
    # Вычисляем наклон MA20 за последние 5 баров (динамика тренда)
    if len(df['ma20'].dropna()) >= 5:
        ma_slope = (df['ma20'].iloc[-1] - df['ma20'].iloc[-5]) / df['ma20'].iloc[-5]
    else:
        ma_slope = 0

    # Определяем режим рынка: тренд, если ADX выше порога, иначе диапазон.
    adx_threshold = 25  # базовый порог, его можно оптимизировать
    regime = 'trend' if current_adx > adx_threshold else 'range'
    
    direction = 'NO SIGNAL'
    atr_multiplier = 1.0
    risk_reward = 1

    # Логика для трендового режима с учетом наклона MA
    if regime == 'trend':
        if current_price > current_ma20 and current_rsi < 50 and current_macd > 0:
            direction = 'LONG'
            atr_multiplier = 1.0
            # Если наклон MA значительный (сильный тренд), используем более агрессивный риск/награда
            risk_reward = 4 if ma_slope > 0.03 else 3
        elif current_price < current_ma20 and current_rsi > 50 and current_macd < 0:
            direction = 'SHORT'
            atr_multiplier = 1.0
            risk_reward = 4 if ma_slope < -0.03 else 3
    else:
        # Режим диапазона – сначала проверяем экстремальные значения RSI
        if current_rsi < 30:
            direction = 'LONG'
            atr_multiplier = 0.5
            risk_reward = 2
        elif current_rsi > 70:
            direction = 'SHORT'
            atr_multiplier = 0.5
            risk_reward = 2
        # Если по RSI сигнал не сформирован, проверяем положение цены относительно Bollinger Bands
        if direction == 'NO SIGNAL':
            margin = 0.01  # допускаем отклонение ~1%
            if current_price <= lower_band * (1 + margin):
                direction = 'LONG'
                atr_multiplier = 0.5
                risk_reward = 2
            elif current_price >= upper_band * (1 - margin):
                direction = 'SHORT'
                atr_multiplier = 0.5
                risk_reward = 2

    trade = {}
    if direction != 'NO SIGNAL' and not pd.isna(current_atr):
        risk_distance = atr_multiplier * current_atr
        if direction == 'LONG':
            stop_loss = current_price - risk_distance
            take_profit = current_price + risk_reward * risk_distance
        else:
            stop_loss = current_price + risk_distance
            take_profit = current_price - risk_reward * risk_distance
        trade = {
            'direction': direction,
            'entry': round(current_price, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'regime': regime,
            'adx': round(current_adx, 2),
            'rsi': round(current_rsi, 2),
            'ma_slope': round(ma_slope, 4),
            'risk_reward': f"1:{risk_reward}"
        }

    return {'symbol': symbol, 'trade': trade}

# =============================================================================
# Основная функция: Получение сигналов и расчёт объёма позиции
# =============================================================================

def main():
    # Начальный депозит (в $) и риск на сделку (например, 2% от депозита)
    deposit = 10.0
    risk_percent = 0.02

    # Подключаемся к бирже Binance USDT‑M (используем публичные данные)
    exchange = ccxt.binanceusdm({'enableRateLimit': True})
    print("Получаем топ-10 пар USDT по объёму...")
    pairs = get_top_pairs(exchange, limit=10)
    results = []

    for symbol in pairs:
        print(f"Обрабатываем {symbol} ...")
        try:
            res = analyze_symbol_improved(exchange, symbol)
            results.append(res)
        except Exception as e:
            print(f"Ошибка при обработке {symbol}: {e}")

    print("\nРезультаты анализа и рекомендации по позициям:")
    for res in results:
        print(f"Парa: {res['symbol']}")
        if res['trade']:
            trade = res['trade']
            pos_size = calculate_position_size(deposit, risk_percent, trade['entry'], trade['stop_loss'])
            print(f"   Режим: {trade['regime']}, ADX: {trade['adx']}, RSI: {trade['rsi']}, MA slope: {trade['ma_slope']}")
            print(f"   Направление: {trade['direction']}")
            print(f"   Цена входа: {trade['entry']}$, Стоп-лосс: {trade['stop_loss']}$, Тейк-профит: {trade['take_profit']}$")
            print(f"   Соотношение риск/награда: {trade['risk_reward']}")
            print(f"   Рекомендуемый объём позиции: {round(pos_size, 4)} единиц")
        else:
            print("   Сигнал не сформирован.")
        print("-" * 40)

if __name__ == "__main__":
    main()
