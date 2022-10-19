import streamlit as st
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor, ceil
from tqdm import tqdm
import gc


da = pd.read_csv('traffic_clean.csv')
da = da.drop('ID', axis=1)
useless_cols1 = ['Quartermedian_vehicles', 'day_of_weekmin_vehicles', 'Quartermax_vehicles', 'Quarter', 'Quartermin_vehicles', 'day_of_weekmedian_vehicles',
                 'Vehicles', 'Seconds', 'Junction']
useless_cols2 = ['Year', 'Quartermax_vehicles', 'Quartermedian_vehicles', 'day_of_weekmedian_vehicles', 'day_of_weekmin_vehicles', 'Quartermin_vehicles', 'Quarter',
                 'Vehicles', 'Seconds', 'Junction']
useless_cols3 = ['Quartermax_vehicles', 'Quarter', 'Monthmax_vehicles', 'Monthmin_vehicles', 'Timemedian_vehicles', 'Quarterstd_vehicles', 'Timemin_vehicles', 'Year', 'Quartermedian_vehicles', 'day_of_weekmedian_vehicles', 'day_of_weekmean_vehicles', 'day_of_weekmin_vehicles', 'day_of_weekmax_vehicles', 'Quartermean_vehicles', 'day_of_monthmin_vehicles', 'Quartermin_vehicles',
                 'Vehicles', 'Seconds', 'Junction']
useless_cols4 = ['Quarter', 'day_of_weekmin_vehicles', 'Monthmin_vehicles', 'day_of_weekmedian_vehicles', 'Year', 'Quartermedian_vehicles', 'Quartermean_vehicles', 'Quartermin_vehicles', 'Quartermax_vehicles', 'day_of_year', 'Quarterstd_vehicles', 'Month', 'Monthmean_vehicles',
                 'Vehicles', 'Seconds', 'Junction']

pickle_in = open("junction1_model.pkl", "rb")
junc1 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("junction2_model.pkl", "rb")
junc2 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("junction3_model.pkl", "rb")
junc3 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("junction4_model.pkl", "rb")
junc4 = pickle.load(pickle_in)
pickle_in.close()

# @app.route('/')


def welcome():
    return "Welcome All"

# @app.route('/predict',methods=["Get"])


def predict_traffic(junction, DateTime):

    df = pd.DataFrame(columns=['DateTime', 'Junction', 'Year', 'Month',
                      'day_of_month', 'day_of_week', 'Date', 'Time', 'day_of_year'])

    if isinstance(DateTime, str):
        df.loc[0] = 0
    else:
        b = len(DateTime)
        df.iloc[:b] = 0
    df['DateTime'] = pd.to_datetime(DateTime)

    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['day_of_month'] = df['DateTime'].dt.day
    df['Junction'] = junction
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['Date'] = df['DateTime'].dt.date
    df['Time'] = df['DateTime'].dt.hour

    df['day_of_year'] = df['DateTime'].dt.dayofyear
    df['Seconds'] = pd.to_timedelta(df['DateTime'].dt.strftime(
        '%H:%M:%S')).dt.total_seconds().astype(int)
    df['DateTime'] = df['DateTime'].values.astype(np.int64) / 10 ** 9

    pkl_file = open('L_encoder.pkl', 'rb')
    L_encoder = pickle.load(pkl_file)
    pkl_file.close()
    df['Date'] = L_encoder.fit_transform(df['Date'])
    a = len(df)

    db = pd.concat([da, df], axis=0)

    def agg_functions(df1):
        features = ['Month', 'Quarter', 'day_of_month',
                    'day_of_week', 'Time', 'day_of_year']
        for x in tqdm(features):
            t = df1.groupby(x)['Vehicles'].agg(
                ['std', 'max', 'min', 'mean', 'median'])
            t.columns = [x+c+'_vehicles' for c in t.columns]
            t = t.astype({c: np.float32 for c in t.columns})
            t.reset_index(inplace=True)
    #         display(t)
            # display(t.T.to_dict('list'))
            df1 = df1.merge(t, on=x, how='left')
            gc.collect()
        return df1
    df = agg_functions(db)
    df = df.tail(a)
    #df = df.drop(useless_cols, axis=1).reset_index(drop=True)

    def to_ceil(array):
        ceiled = []
        for i in range(len(array)):
            d = ceil(array[i])
            ceiled.append(d)
        return ceiled

    if junction == 1:
        df = df.drop(useless_cols1, axis=1).reset_index(drop=True)
        prediction = junc1.predict(df)
        prediction = to_ceil(prediction)
        if isinstance(DateTime, str):
            predictions = prediction
            for i in predictions:
                predictions = i
        else:
            predictions = pd.DataFrame(columns=['Vehicle Number Predictions'])
            predictions['Vehicle Number Predictions'] = prediction
    elif junction == 2:
        df = df.drop(useless_cols2, axis=1).reset_index(drop=True)
        prediction = junc2.predict(df)
        prediction = to_ceil(prediction)
        if isinstance(DateTime, str):
            predictions = prediction
            for i in predictions:
                predictions = i
        else:
            predictions = pd.DataFrame(columns=['Vehicle Number Predictions'])
            predictions['Vehicle Number Predictions'] = prediction
    elif junction == 3:
        df = df.drop(useless_cols3, axis=1).reset_index(drop=True)
        prediction = junc3.predict(df)
        prediction = to_ceil(prediction)
        if isinstance(DateTime, str):
            predictions = prediction
            for i in predictions:
                predictions = i
        else:
            predictions = pd.DataFrame(columns=['Vehicle Number Predictions'])
            predictions['Vehicle Number Predictions'] = prediction
    elif junction == 4:
        df = df.drop(useless_cols4, axis=1).reset_index(drop=True)
        prediction = junc4.predict(df)
        prediction = to_ceil(prediction)
        if isinstance(DateTime, str):
            predictions = prediction
            for i in predictions:
                predictions = i
        else:
            predictions = pd.DataFrame(columns=['Vehicle Number Predictions'])
            predictions['Vehicle Number Predictions'] = prediction

    return predictions


def main():
    #st.title("Junction Traffic Predictor by Team Scipy")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;font-family:'Caveat',cursive;font-weight: 400;max-width: 800px; width: 85%; margin: 0 auto;">Junction Traffic Predictor</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    date = st.sidebar.date_input(
        'Date', datetime.datetime.today())  # (2011, 1, 28))
    time = st.sidebar.time_input(
        'Time', datetime.datetime.now())  # (hour=18, minute=54, second=30))
    datestr = date.strftime("%Y-%m-%d")
    timestr = time.strftime("%H:%M:%S")
    DateTime = datestr + ' ' + timestr
    DateTime1 = pd.to_datetime(DateTime)
    junction = st.number_input(
        'Choose Junction:', min_value=1, max_value=4, value=1, step=1, format='%d')

    prediction = predict_traffic(junction, DateTime)

    result = ""
    if st.button("Predict"):
        result = prediction
    st.success('Successful!!!')
    st.write('The Traffic Prediction for Junction', junction, ' at Date:',
             date, 'and Time:', time, 'is: ', prediction, '\u00B1 3 Vehicles')
    st.write('OR')

    with st.expander("Upload CSV with DateTime Column"):
        st.write("IMPORT DATA")
        st.write(
            "Import the time series CSV file. It should have one column labelled as 'DateTime'"
        )
        data = st.file_uploader("Upload here", type="csv")
        st.session_state.counter = 0
        if data is not None:
            dataset = pd.read_csv(data)
            dataset["DateTime"] = pd.to_datetime(dataset["DateTime"])
            dataset = dataset.sort_values("DateTime")

            junction = st.number_input(
                "Which Junction:", min_value=1, max_value=4, value=1, step=1, format="%d", key='hdth2573@%#dgjsj@'
            )

            results = predict_traffic(junction, dataset["DateTime"])
            st.write("Upload Sucessful")
            st.session_state.counter += 1
            if st.button("Predict Dataset"):
                result = results
                result = pd.concat([dataset, result], axis=1)
                st.success("Successful!!!")
                st.write("Predicting for Junction", junction)
                resulta = result.copy()
                resulta['DateTime'] = resulta['DateTime'].astype(str)
                st.write(resulta)

                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv(index=False).encode("utf-8")

                csv = convert_df(result)
                st.download_button(
                    label="Download Traffic Predictions as CSV",
                    data=csv,
                    file_name="Traffic Predictions.csv",
                    mime="text/csv",
                )
                fig = plt.figure(figsize=(12, 10))
                sns.lineplot(
                    x='DateTime', y='Vehicle Number Predictions', data=result)

                st.write("The following plot shows predicted Vehicle numbers at Junction",
                         junction, "for your provide Datetime Frame:")
                st.pyplot(fig)
                st.session_state.counter += 1

    with st.expander("Real Time Forecasts with Datetime Range"):
        st.write('From:')
        date1 = st.date_input(
            'Date', datetime.date(2017, 7, 1), key='hst%N@&n8&dn2')  # (2011, 1, 28))
        time1 = st.time_input(
            'Time', datetime.time(0, 00), key='hsye^8nyBT@8b2')  # (hour=18, minute=54, second=30))
        datestr = date1.strftime("%Y-%m-%d")
        timestr = time1.strftime("%H:%M:%S")
        DateTime = datestr + ' ' + timestr
        st.write('To:')
        date2 = st.date_input(
            'Date', datetime.datetime.today(), key='dn&@T6thSGSJ6t5T')  # (2011, 1, 28))
        time2 = st.time_input(
            'Time', datetime.datetime.now(), key='HGt73n7bgs6Jsyu&#5$@nysh')  # (hour=18, minute=54, second=30))
        datestr = date2.strftime("%Y-%m-%d")
        timestr = time2.strftime("%H:%M:%S")
        DateTime1 = datestr + ' ' + timestr
        #DateTime1 = pd.to_datetime(DateTime)
        st.write('Real Time Forecasts')
        junction = st.number_input(
            'Choose Junction:', min_value=1, max_value=4, value=1, step=1, format='%d', key='ksu2@uNnyw1*2')
        forecast_junc = pd.date_range(
            start=DateTime, end=DateTime1, freq='H')
        forecast_junc = pd.DataFrame({'DateTime': forecast_junc})
        # if st.button('Forecast'):

        st.write('Real Time Forecast for Junction',
                 junction, 'from', DateTime, 'to', DateTime1)
        forecast = predict_traffic(junction, forecast_junc['DateTime'])
        forecast = pd.concat([forecast_junc, forecast], axis=1)
        # st.write(forecast_junc)

        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode("utf-8")

        csv = convert_df(forecast)
        st.download_button(
            label="Download DateTime Range Predictions as CSV",
            data=csv,
            file_name="Traffic Predictions by DateTime Range.csv",
            mime="text/csv",
        )
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot(
            x='DateTime', y='Vehicle Number Predictions', data=forecast)
        st.pyplot(fig)

        st.text("Team Scipy")
        st.text("Hamoye Premiere Project")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()
