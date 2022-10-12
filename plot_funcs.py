import pandas as pd
import numpy as np
from itertools import groupby
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress


def fractal_pivots_df(df, n1, n2):
    def _pivotid(df1, l, n1, n2):  # n1 n2 before and after candle l
        if l - n1 < 0 or l + n2 >= len(df1):
            return 0

        pividlow = 1
        pividhigh = 1
        for i in range(l - n1, l + n2 + 1):
            if df1.low[l] > df1.low[i]:
                pividlow = 0
            if df1.high[l] < df1.high[i]:
                pividhigh = 0
        if pividlow and pividhigh:
            return 3
        elif pividlow:
            return 1
        elif pividhigh:
            return 2
        else:
            return 0

    def _pointpos(x):
        if x["pivot"] == 1:
            return x["low"] - 1e-3
        elif x["pivot"] == 2:
            return x["high"] + 1e-3
        else:
            return np.nan

    df["pivot"] = df.apply(lambda x: _pivotid(df, x.name, n1, n2), axis=1)
    df["pointpos"] = df.apply(lambda row: _pointpos(row), axis=1)

    return df


def window_shift(df, window_size, window_shift=5):

    # to make sure the new level area does not exist already
    def is_far_from_level(value, levels, df):
        ave = np.mean(df["high"] - df["low"])
        return np.sum([abs(value - level) < ave for _, level in levels]) == 0

    pivots = []
    max_list = []
    min_list = []
    for i in range(window_size, len(df) - window_size):
        # taking a window of 9 candles
        high_range = df["high"][i - window_size : i + window_size]
        current_max = high_range.max()
        # if we find a new maximum value, empty the max_list
        if current_max not in max_list:
            max_list = []
        max_list.append(current_max)
        # if the maximum value remains the same after shifting 5 times
        if len(max_list) == window_shift and is_far_from_level(current_max, pivots, df):
            pivots.append((high_range.idxmax(), current_max))

        low_range = df["low"][i - 5 : i + 5]
        current_min = low_range.min()
        if current_min not in min_list:
            min_list = []
        min_list.append(current_min)
        if len(min_list) == 5 and is_far_from_level(current_min, pivots, df):
            pivots.append((low_range.idxmin(), current_min))
    return pivots


def _get_breakout_point(df, minimal_continuity=3):
    if len(df) != 0:
        gb = groupby(enumerate(df.index.tolist()), key=lambda x: x[0] - x[1])
        all_groups = ([i[1] for i in g] for _, g in gb)
        candidates = list(filter(lambda x: len(x) >= minimal_continuity, all_groups))

        if len(candidates) != 0:
            return int(candidates[0][0])
        else:
            return "NaN"
    else:
        return "NaN"


def inspect_interval(candle_df, intervals, minimal_continuity):
    """
    Interval should be a list of two values indicating the boundaries

    """

    def _find_point(breakout_df):
        down_df = breakout_df[breakout_df.close < breakout_df.min_bound]
        up_df = breakout_df[breakout_df.close > (breakout_df.max_bound)]
        breakdown_point = _get_breakout_point(down_df, minimal_continuity)
        breakup_point = _get_breakout_point(up_df, minimal_continuity)
        return breakup_point, breakdown_point

    result_lists = list()
    for interval in intervals:
        t_df = candle_df.loc[interval[0] : interval[1]][["high", "low"]].dropna()
        x = t_df.index
        minim = t_df.low
        maxim = t_df.high
        slmin, intercmin, rmin, pmin, semin = linregress(x, minim)
        slmax, intercmax, rmax, pmax, semax = linregress(x, maxim)
        X = list(range(interval[1] + 1, len(candle_df)))
        min_bound = pd.Series(slmin * np.array(X) + intercmin, index=X)
        max_bound = pd.Series(slmax * np.array(X) + intercmax, index=X)
        breakout_df = candle_df.iloc[interval[1] :]
        breakout_df.loc[:, "min_bound"] = min_bound.copy()
        breakout_df.loc[:, "max_bound"] = max_bound.copy()
        breakout_df = breakout_df[(breakout_df["pivot"] != 3)]
        breakup_point, breakdown_point = _find_point(breakout_df)
        # [for point in [breakup_point, breakdown_point] if type(point)==int ]
        #  breakout_df.loc[interval[0]].time,
        # breakout_df.loc[interval[1]].time
        result_lists.append(
            interval
            + [
                breakup_point,
                breakdown_point,
                slmin,
                intercmin,
                rmin,
                pmin,
                semin,
                slmax,
                intercmax,
                rmax,
                pmax,
                semax,
            ]
        )

    result_df = pd.DataFrame(result_lists)
    result_df.columns = [
        "interval_start",
        "interval_end",
        "breakup_point",
        "breakdown_point",
        "slmin",
        "intercmin",
        "rmin",
        "pmin",
        "semin",
        "slmax",
        "intercmax",
        "rmax",
        "pmax",
        "semax",
    ]
    return result_df


def get_candlestick_chart(df, begin, end, frequency):
    df = df[(df.index > begin) & (df.index <= end)]
    candle_df = pd.DataFrame()
    candle_df["open"] = df.resample(frequency).first()
    candle_df["high"] = df.resample(frequency).max()
    candle_df["low"] = df.resample(frequency).min()
    candle_df["close"] = df.resample(frequency).last()
    candle_df = candle_df.reset_index()
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=candle_df.time,
                open=candle_df["open"],
                high=candle_df["high"],
                low=candle_df["low"],
                close=candle_df["close"],
            )
        ]
    )
    fig.update_layout(xaxis_rangeslider_visible=False, height=800)

    return fig, candle_df


def plot_breakout(candle_df, result_df, point_index, fig):
    record = result_df.loc[point_index]
    up_point = record.breakup_point
    down_point = record.breakdown_point
    fig.add_vrect(
        x0=candle_df.loc[record.interval_start].time,
        x1=candle_df.loc[record.interval_end].time,
        annotation_text="Interval",
        annotation_position="top left",
        fillcolor="yellow",
        opacity=0.25,
        line_width=0,
    )

    if down_point != "NaN":
        x_range = range(record.interval_start, down_point + 1)
        x_axis = candle_df.time.iloc[record.interval_start : down_point + 1]
        color = "green"
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=record.slmin * np.array(x_range) + record.intercmin,
                mode="lines",
                marker_color=color,
            )
        )
        fig.add_scatter(
            x=candle_df.time.iloc[down_point : down_point + 1],
            y=[candle_df.iloc[down_point].low],
            mode="markers",
            marker=dict(size=10, color=color),
            name="breakdown_point",
        )
    if up_point != "NaN":
        x_range = range(record.interval_start, up_point + 1)
        x_axis = candle_df.time.iloc[record.interval_start : up_point + 1]
        color = "blue"
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=record.slmax * np.array(x_range) + record.intercmax,
                mode="lines",
                marker_color=color,
            )
        )
        fig.add_scatter(
            x=candle_df.time.iloc[up_point : up_point + 1],
            y=[candle_df.iloc[up_point].high],
            mode="markers",
            marker=dict(size=10, color=color),
            name="breakup_point",
        )
