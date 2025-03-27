import numpy as np
import pandas as pd


def data_to_dict(filename):
    df = pd.read_csv(filename, delimiter=";", parse_dates=[2], date_format="%Y-%m-%d")

    df = df.rename(columns={'Date_measurement':'Date', 'RWZI_AWZI_name':'Location', 'RNA_flow_per_100000':'Observation'}, inplace=False)

    df = df[['Date', 'Location', 'Observation']]

    dates = df['Date'].dt.strftime("%Y-%m-%d").drop_duplicates().to_list()
    start_date = dates[0]

    df['Date'] = (df['Date'] - df['Date'].min()).dt.days


    df_grouped = df.pivot(index='Date', columns='Location', values='Observation')
    locations = df_grouped.axes[1].to_list()
    days = df_grouped.axes[0].to_list()

    df_grouped = df_grouped.fillna(-1)

    data = df_grouped.to_numpy(na_value=np.nan)

    data_dict = {
        "Data": data,
        "Days": days,
        "Dates": dates,
        "Start date": start_date,
        "Locations": locations
    }

    return data_dict




def get_loc_data(data_dict, index):

    data = data_dict["Data"]
    days = data_dict["Days"]
    dates = data_dict["Dates"]
    locations = data_dict["Locations"]

    loc = locations[index]
    loc_data = data[:,index]


    loc_days = np.array([days[date] for date, val in enumerate(loc_data) if val!=-1])
    loc_dates = np.array([dates[date] for date, val in enumerate(loc_data) if val!=-1])

    temp = loc_data[loc_data != -1]

    loc_data = np.array([loc_days, temp])


    return loc, loc_data, loc_days, loc_dates




def day_index_to_date(day_indices, start_date, list=False):

    if not list:
        day_indices = [day_indices]

    dates = [pd.to_datetime(start_date, format="%Y-%m-%d") + pd.Timedelta(days=i) for i in day_indices]

    if list:
        return [date.strftime("%Y-%m-%d") for date in dates]
    else:
        return [date.strftime("%Y-%m-%d") for date in dates][0]


def date_to_day_index(dates, start_date, list=False):

    if not list:
        dates = [dates]

    day_indices = [pd.to_datetime(i, format="%Y-%m-%d") - pd.to_datetime(start_date, format="%Y-%m-%d") for i in dates]

    if list:
        return [index.days for index in day_indices]
    else:
        return [index.days for index in day_indices][0]
    







