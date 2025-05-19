from datetime import datetime
from pytz import UTC
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def time_converter(given_time: str) -> float:
    # converts the given time into a datetime object
    datetime_obj = datetime.strptime(given_time, "%Y-%m-%d %H:%M:%S")
    # sets the timezone of the datetime object to UTC
    datetime_obj_utc = datetime_obj.replace(tzinfo=UTC)
    # calculates the difference between the datetime object and the Unix Epoch time in milliseconds
    milliseconds = (datetime_obj_utc.timestamp() - datetime(1970, 1, 1, tzinfo=UTC).timestamp()) * 1000
    return milliseconds
    
def convert_time_in_file() -> None:
    # load csv files
    data_COM5 = np.genfromtxt('data/data_COM5.csv', delimiter=';', skip_header=True, dtype=str)
    data_COM7 = np.genfromtxt('data/data_COM7.csv', delimiter=';', skip_header=True, dtype=str)
    # create new csv files
    new_data_COM5 = np.zeros((data_COM5.shape[0], 2))
    new_data_COM7 = np.zeros((data_COM7.shape[0], 2))
    # convert time
    for i in range(0, data_COM5.shape[0]):
        new_data_COM5[i, 0] = time_converter(data_COM5[i, 9])
        new_data_COM5[i, 1] = data_COM5[i, 4]
    for i in range(0, data_COM7.shape[0]):
        new_data_COM7[i, 0] = time_converter(data_COM7[i, 9])
        new_data_COM7[i, 1] = data_COM7[i, 4]
    # save new csv files
    np.savetxt("data/ss.csv", new_data_COM5, delimiter=";")
    np.savetxt("data/a.csv", new_data_COM7, delimiter=";")

def interpolate(data):
    # create new array with same length as data
    new_data = np.arange(data.shape[0])
    # interpolate
    nan_mask = np.where(np.isfinite(data))
    interpolated = interp1d(new_data[nan_mask], data[nan_mask], bounds_error=False, kind = 'cubic')
    interpolated = np.where(np.isfinite(data), data, interpolated(new_data))    
    return interpolated

def trend(data: list, degree: int) -> list:
    # data
    t = data[:,0]
    d = data[:,1]
    # determine trend
    param = np.polyfit(t, d, degree)
    trend = np.polyval(param, t)
    return trend

def moving_average(distances: list, half_window_size: int) -> list:
    # start moving average
    moving_averages = [distances[i] for i in range(0, half_window_size)]
    for i in range(half_window_size, len(distances) - half_window_size):
        moving_averages.append(np.mean(distances[i - half_window_size : i + half_window_size]))
    moving_averages += [distances[i] for i in range(len(distances) - half_window_size, len(distances))]
    return np.array(moving_averages)

def data_gaps(timestamps: list) -> list:
    # data
    gaps = []
    # determine gaps
    for i in range(1, len(timestamps)):
        gaps.append(timestamps[i] - timestamps[i-1])
    # storing & returning
    return gaps

def limits(data: list, percentage: float) -> list:
    mean = np.mean(data)
    percentage = percentage / 100
    upper_limit = mean * (1+percentage)
    lower_limit = mean * (1-percentage)
    return [upper_limit, lower_limit]

def residuals(distances: list, trend: list) -> list:
    # residuals
    residuals = distances - trend
    return residuals

def delta_t(timestamps: list) -> list:
    # delta_t
    delta_t = []
    for i in range(1, len(timestamps)):
        delta_t.append(timestamps[i] - timestamps[i-1])
    dt = np.mean(delta_t)
    return dt

def outliers(residuals: list, upper_limit: float, lower_limit: float) -> list:
    # outliers
    outliers = []
    for i in range(0, len(residuals)):
        if residuals[i] > upper_limit or residuals[i] < lower_limit:
            outliers.append(i)
    return outliers

def detect_peaks(y):
    # Find local maxima
    peaks, _ = find_peaks(y)
    return peaks




if __name__ == "__main__":
    pass



