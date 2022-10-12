import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from load_data import load_data
from plot_funcs import (
    get_candlestick_chart,
    fractal_pivots_df,
    inspect_interval,
    plot_breakout,
    window_shift,
)

# from breakout_detection import fractal_pivots_df
import datetime


# Load data
df = load_data("data/PriceData.pickle")
st.set_page_config(
    layout="wide",
)

###################################### Side Bar ###########################################
with st.sidebar:
    start_date = st.date_input("Start date", datetime.date(2022, 1, 1))
    end_date = st.date_input("End date", datetime.date(2022, 7, 15))
    st.write("Frequency")
    col1, col2, col3 = st.columns(3)
    with col1:
        day = st.number_input("Days", value=1, step=1)
    with col2:
        hour = st.number_input("Hours", step=1)
    with col3:
        minutes = st.number_input("Mins", step=1)

    frequency = "%sD%sH%sMIN" % (int(day), int(hour), int(minutes))

    st.write("Detect Pivot Points")
    with st.expander("Fractal Points", expanded=True):
        with st.container():
            fractal_win_size = st.slider(
                "Window Size -number of candle sticks before and after the pivot points",
                0,
                100,
                3,
            )
            candle_distance = st.slider(
                "Candle distance - the minimal interval between pivots points",
                0,
                50,
                5,
            )
            fractal_plot = st.checkbox("Plot Pivot Points in Purple")
            plot_interval = st.checkbox("Plot Interval Lines")

    with st.expander("Window Shift Points", expanded=True):
        with st.container():
            ws_win_size = st.slider("Window Size", 0, 100, 3)
            shift = st.slider("Number of shifts", 0, 20, 5)
            plot_level = st.checkbox("Plot Support/Resistance Levels")

    st.write("Detect Beakout Point")
    with st.expander("detect breakout"):
        with st.container():
            minimal_continuity = st.slider(
                "minimal continuity - minimal number of candles showing the same trend",
                0,
                20,
                3,
            )
            show_table = st.checkbox("Display Breakout Candidates")


st.title("Price Breakout Point Detection")

###################################### Main Frame ###########################################
st.header("Observe Price Data")
with st.expander("ℹ️ - Instructions"):
    st.write(
        # "This app could help you observe the price pattern of given time range with specified frequency and automatically detect breakout points
        # "
        """
        - Visualise the candlestick by setting the timeframe (between 2022-01-01 to 2022-07-15) and **Frequency** on the left side bar. 
        - Add **Pivot Points**:
          * Pivot points are determined by **fractal**, a candlestick pattern consisting of a few candles with the one in the middle having the lowest or highest values. A defined number of candles within a **window** the the middle candle should displaying an opposite up or down trend. 
          * You can play with the following parameters to find good pivot points: 
            - **Window Size**: the number of candles on each side of your pivot candle. The window size determines how sparse or intensive your pivot points will be. The larger the window, the fewer pivot points we will get from the timeframe. _Note: **Frequency** of the candlestick should be compatible with window size. For example, a 15-minute candlestick chart will require a larger value of one-side window size (e.g. 100) than daily chart whose windowsize could be 3-5._ 

            - **Candle Distance**: this defines the minimal distance between two candles. The larger distance could help your remove some pivot points too close to each other
          * **Pivot points and interval lines** can be plotted on the chart when you tick the according boxes
            (_Intervals are used in the later process of breakout detection_)
          * Pivot points can also be found with window shift methods.This method is similar to the fractal method except that the window can shift a few times. The pivot points should be the maximum or minimum values across all windows. The additional parameter you should set is:
             - **Number of Shift**: times of shift of a given window size
        - **Support and Resistance Levels** can be infered from the pivot points and you can plot them by ticking 'Plot Support/Resistance Levels' box
         
        
        
        """
    )
plot_df = df[["time", "processed_price"]].set_index("time")
fig1, candle_df = get_candlestick_chart(
    plot_df, str(start_date), str(end_date), frequency
)
fig1.update_layout(
    title_text="Candlestick Chart from %s to %s at frequency of %s"
    % (str(start_date), str(end_date), frequency),
    font_size=10,
)


# Fractal Pivot points
candle_df = fractal_pivots_df(candle_df, n1=fractal_win_size, n2=fractal_win_size)
pos_df = candle_df[~(candle_df.pointpos.isnull())].reset_index()
s = pos_df["index"].diff(1)
target_points = sorted(
    pos_df.iloc[s[s > candle_distance].index]["index"].tolist()
    + [candle_df.index.min(), candle_df.index.max()]
)
intervals = [
    [target_points[i], target_points[i + 1]] for i in range(len(target_points) - 1)
]
if fractal_plot:
    fig1.add_scatter(
        x=candle_df.time,
        y=candle_df["pointpos"],
        mode="markers",
        marker=dict(size=10, color="MediumPurple"),
        name="fractal_pivot",
    )


if plot_interval:
    for t_point in target_points[1:-1]:
        fig1.add_vline(x=candle_df.iloc[t_point].time, line_width=1, line_dash="dash")

if plot_level:
    points_list = window_shift(candle_df, window_size=ws_win_size, window_shift=shift)
    for point in points_list:
        fig1.add_shape(
            type="line",
            x0=candle_df.loc[point[0]].time,
            y0=point[1],
            x1=candle_df.time.max(),
            y1=point[1],
            line_width=1,
            line_dash="dash",
            line_color="Blue",
        )

st.plotly_chart(fig1, use_container_width=True)

st.header("Detect Price Pattern with Breakout Points")
with st.expander("ℹ️ - Instructions"):
    st.write(
        """
        1. **Range/Pattern Detection**
        - The program can automatically detect *sets of minimal and maximal boundaries*, which could either form:
          * **Support or Resistance Levels** or
          * **Chart Patterns** like triangle or flag 

        - The min and max boundaries are two lines fitted by low and high price data points respectively within the interval
          * ** Linear Regressions are used to fit the lines with data points between two interval boundaries
          * **Pivot points** are enssential to determine the intervals thus you should pay attention to parameters like window size
        
        2. **Breckout Detection** 
        - Breakout points are determined by comparing the actual close price after the interval boundary with **predicted** min or max by the linear regression fitted within the interval.  The successful candidates for breakout points are:
          * **Breakdown** is the first candle when close price lower than predicted minimal while **Breakdown** is the first candle when close price higher than predicted maximal
          * **Minimal Contnuity** which is the minimal sequence of consecutive candles fitting the requirement. This parameter is effective in avoiding *Fake Breakout*
        - To check the detection result, you can tick the 'Display Candidate Breakout' and you can see a result table with sets of patterns detected and their associated breakout points:
          * **Interval Timeframe** used to determine the max/min boundary lines
          * **Detected Breakout Points** assocated with the 
          * **Coefficient of linear regressions**:
              * _slope_: slmin, slmax
              * _intercept_: intercmin, intermax
          * **Statistical significance of liner regressions**: 
              * _peason r_: rmin, rmax
              * _p value_: pmin, pmax
              * _standard error_: semin,semax  
        3 **Select and Inspect Breakouts 
        - You can use these information to **select** a few breakout points for further inspections. 
          * For example,
              * **Shape of the Pattern** could be indicated by **the signs (+/-) and the values of the slopes of maximum and minimal boundaries**
              * The **regression statistics (e.g. r, p-value)** can help us filter out some badly fitted boundaries. Thus removing noise lines. 
        - Eventually, you can visualise the selected patterns and detect breakup (Blue) or breakdown(Green) points on the chart
             * Simply put the **index** (the left most column on teh table) into the multiselect bar to plot the pattern and breakout in interest 
           
         """
    )
if show_table:
    breakout_result = inspect_interval(candle_df, intervals, minimal_continuity)
    display_df = breakout_result.copy()
    for col in breakout_result.columns.tolist()[:4]:
        display_df[col] = display_df[col].apply(
            lambda x: candle_df.loc[int(x)].time if x != "NaN" else x
        )
    st.dataframe(display_df, use_container_width=True)
    st.header("Inspect Selected Patterns and Beakout Points")
    options_index = st.multiselect(
        "Select Patterns to Inspect", display_df.index.tolist()
    )

    for point_index in options_index:
        plot_breakout(candle_df, breakout_result, point_index, fig1)

    st.plotly_chart(fig1, use_container_width=True)
