import urllib.request
from gather_data import load_all_posts
from gather_data import save_list
from gather_data import load_list
from gather_data import fetch_code_from_url
from geopy.geocoders import Nominatim
from api_keys import geopy_loc_app_name
import numpy as np
import pandas as pd
import os
import time
import datetime as dt
import matplotlib.pyplot as plt

geolocator = Nominatim(user_agent=geopy_loc_app_name)



def get_unique_cities(location="998_generated\\"):
    """
    A function to get all the unique cities with for all the posts.

    Parameters
    ----------
    location : str, optional
        The relative path to the data folder. Default is "998_generated\\".

    Returns
    -------
    cities : list of str
        All the unique cities.
    """
    posts = load_all_posts(location = location)
    cities = []
    for post in posts:
        if post.city in cities:
            pass
        else:
            cities.append(post.city)

    return cities


def get_missing_cities(data_location="998_generated\\",
                       weather_location="997_weather_data\\"):
    """
    A function to get all the unique cities for all the posts
    that are not downlaoded yet.

    Parameters
    ----------
    data_location : str, optional
        The relative path to the data folder. 
        Default is "998_generated\\".
    weather_location : str, optional
        The relative path to the weather data folder. 
        Default is "997_weather_data\\".

    Returns
    -------
    cities_needed : list of str
        All the unique citie that are still missing in among the weather data.

    """
    cities_needed = get_unique_cities(location = data_location)
    weather_files_downlaoded = os.listdir(weather_location)
    for file in weather_files_downlaoded:
        cities_needed.remove(file[:-4])

    return cities_needed



def get_GPS(city,
            country="Slovakia"):
    """
    A simple wrapper for getting GPS coordinates based on city, country 
    using geopy.

    Parameters
    ----------
    city : str
        The city of interest.
    country : str, optional
        The country of interest. The default is "Slovakia".

    Returns
    -------
    latitude : float
        The geographic latitude in degrees decimals, positive is North.
    longitude : float
        The geographic longitude in degrees decimals, positive is East.

    """
    address = geolocator.geocode(city + ", "+country)
    return (address.latitude, address.longitude)
    

def build_gps_database(cities,
                       location="996_gps\\"):
    """
    A function to build a local database of geolocations, so that we do not 
    have to wait for geopy to retunr those every run.

    Parameters
    ----------
    cities : list of str
        The cities of interest.
    location : str, optional
        The relative path to the data folder. Default is "996_gps\\".

    Returns
    -------
    data : list of lists
        data[0] is cities. data[1:2] are np.arrays of longitudes and latitudes.
    """
    cities = get_unique_cities()   
    x, y = np.zeros(0), np.zeros(0)
    for city in cities:
        GPS = get_GPS(city)
        x = np.append(x,GPS[0])
        y = np.append(y,GPS[1])
    data = [cities,x,y]
    save_list(data,"gps_database.pkl",location=location)
    return data
    
    
def fetch_GPS(cities,
              country="Slovakia",
              gps_data_location="996_gps\\"):
    """
    A function that return the locations of the provided cities 
    within a provided country. It will attempt to load the locations from
    a local database, if possible. Otherwise, th locations will be loaded
    using geopy and the wrapper get_GPS().

    Parameters
    ----------
    cities : list of str
        The cities of interest.
    country : TYPE, optional
        DESCRIPTION. The default is "Slovakia".
    gps_data_location : str, optional
        The relative path to the GPS data folder. Default is "996_gps\\".

    Returns
    -------
    x : np.array of float
        Latitudes of the cities.
    y : np.array of float
        Longitudes of the cities.
    """
    x, y = np.zeros(0), np.zeros(0)
    try:
        data = load_list("gps_database.pkl",location=gps_data_location)
    except:
        for city in cities:
            GPS = get_GPS(city)
            x = np.append(x,GPS[0])
            y = np.append(y,GPS[1])
    else:
        for city in cities:
            if city in data[0]:
                index = data[0].index(city)
                x = np.append(x, data[1][index])
                y = np.append(y, data[2][index])
            else:
                GPS = get_GPS(city)
                x = np.append(x, GPS[0])
                y = np.append(y, GPS[1])
    finally:
        return x, y
    
    
def print_map(cities):
    """
    A sunction to print the map of all the provided cities.

    Parameters
    ----------
    cities : list of str
        The cities of interest.

    Returns
    -------
    None.

    """
    x, y = fetch_GPS(cities)
    fig, ax = plt.subplots()
    ax.scatter(y,x,color="red")
    ax.set_aspect(1/np.cos(48.8/180*3.14159))
    ax.set_ylabel("GPS N [deg]")
    ax.set_xlabel("GPS E [deg]")
    plt.show()


def weather_url(latitude,
                longitude,
                start_date="2004-12-01",
                end_date="2023-09-19"):
    """
    To construct the proper URL of open-meteo.

    Example: https://archive-api.open-meteo.com/v1/
             archive?latitude=48.7139&longitude=21.2581
             &start_date=2020-07-26&end_date=2023-09-21
             &daily=temperature_2m_max,temperature_2m_min,
             temperature_2m_mean,precipitation_sum,rain_sum,
             snowfall_sum,precipitation_hours,
             et0_fao_evapotranspiration&timezone=Europe%2FBerlin


    Parameters
    ----------
    latitude : float
        in deg decimal, positive is North.
    longitude : int
        in deg decima, positive is East.
    start_date : str, optional
        String of YYYY-MM-DD, first day of data. The default is "2004-12-01",
        which predates the shroom growing posts archive.
    end_date : str, optional
        String of YYYY-MM-DD, last day of data. The default is "2023-09-19",
        which postdates the shroom growing posts archive.

    Returns
    -------
    str
        The URL.

    """
    return "https://archive-api.open-meteo.com/v1/archive?"+ \
            "latitude="+str(latitude)+ \
            "&longitude="+str(longitude)+ \
            "&start_date="+start_date+ \
            "&end_date="+end_date+ \
            "&daily=temperature_2m_max,temperature_2m_min,"+ \
            "temperature_2m_mean,precipitation_sum,rain_sum,"+ \
            "snowfall_sum,precipitation_hours,"+ \
            "et0_fao_evapotranspiration&timezone=Europe%2FBerlin"


def datetime_from_jsondate(jsondate):
    """
    A function to parse a readable date in json and to produce the datetime.

    Parameters
    ----------
    jsondate : str
        Readable date in the "YYYY-MM-DD" format.

    Returns
    -------
    date : datetime.datetime()
        The same date, time 00:00.

    """
    YYYY = int(jsondate[0:4])
    MM = int(jsondate[5:7])
    DD = int(jsondate[8:10])
    date = dt.datetime(YYYY,MM,DD)
    return date


def build_weather_dataframe(city,
                            country="Slovakia",
                            start_date="2004-12-01",
                            end_date="2023-09-19"):
    """
    A function to build a dataframe of historical daily weather data for a 
    given city between two dates, inclusive.

    Parameters
    ----------
    city : str
        The city of interest.
    country : str
        The country of the city. The default is "Slovakia".
    start_date : str, optional
        String of YYYY-MM-DD, first day of data. The default is "2004-12-01",
        which predates the shroom growing posts archive.
    end_date : str, optional
        String of YYYY-MM-DD, last day of data. The default is "2023-09-19",
        which postdates the shroom growing posts archive.

    Returns
    -------
    data : pd.dataframe
        The dataframe of the weather daily data.

    """
    lat, lon = fetch_GPS([city],country = country)
    json = fetch_code_from_url(weather_url(lat[0],lon[0]))
    df = pd.read_json(json)

    elevation = np.array([df["elevation"]["time"]]*len(df["daily"]["time"]))
    date = np.zeros(0,dtype=dt.datetime)
    for time in df["daily"]["time"]:
        date = np.append(date,datetime_from_jsondate(time))
    temperature_2m_max_C = np.array(df["daily"]["temperature_2m_max"])
    temperature_2m_min_C = np.array(df["daily"]["temperature_2m_min"])
    temperature_2m_mean_C = np.array(df["daily"]["temperature_2m_mean"])
    precipitation_sum_mm = np.array(df["daily"]["precipitation_sum"])
    rain_sum_mm = np.array(df["daily"]["rain_sum"])
    snowfall_sum_cm = np.array(df["daily"]["snowfall_sum"])
    precipitation_hours_h = np.array(df["daily"]["precipitation_hours"])
    evapotranspiration = np.array(df["daily"]["et0_fao_evapotranspiration"])

    data = pd.DataFrame({'date': date,
                         'elevation': elevation,
                         'temperature_2m_max_C': temperature_2m_max_C,
                         'temperature_2m_min_C': temperature_2m_min_C,
                         'temperature_2m_mean_C': temperature_2m_mean_C,
                         'precipitation_sum_mm': precipitation_sum_mm,
                         'rain_sum_mm': rain_sum_mm,
                         'snowfall_sum_cm': snowfall_sum_cm,
                         'precipitation_hours_h': precipitation_hours_h,
                         'evapotranspiration': evapotranspiration})

    return data


def build_weather_database(cities,
                           location="997_weather_data\\"):
    """
    A procedure to save all the wather data. Build like:
        >>> build_weather_database(get_missing_cities()[:2])

    Parameters
    ----------
    cities : list of str
        The cities of interest.
    location : str, optional
        The relative path to the weather data folder. 
        The default is "997_weather_data\\".
    
    Returns
    -------
    none
    """
    n=0
    for city in cities:
        weather_df = build_weather_dataframe(city)
        lat, lon = fetch_GPS([city])
        data = [city,lat[0],lon[0],weather_df]
        save_list(data,city+".pkl",location=location)
        n+=1
        #time.sleep(20)
    print("Saved "+str(n)+" weather database files.")


def load_weather(city,
                 datetime,
                 attributes = ['temperature_2m_mean_C'],
                 location="997_weather_data\\"):
    """
    Loads the weather data for a given day for a given city from the database.

    Parameters
    ----------
    city : str
        The city of interest.

    datetime : dt.datetime or np.datetime64
        The date of interest.

    attributes: list of str, optional
        The attributes of interest. The default is 'temperature_2m_mean_C'.
        Allowed attributes: ['elevation',
                             'temperature_2m_max_C',
                             'temperature_2m_min_C',
                             'temperature_2m_mean_C',
                             'precipitation_sum_mm',
                             'rain_sum_mm',
                             'snowfall_sum_cm',
                             'precipitation_hours_h',
                             'evapotranspiration'].
    
    location : str, optional
        The relative path to the weather data folder. 
        The default is "997_weather_data\\".

    Returns
    -------
    state : list of float
        The values of the requested attributes for the requested datetime.

    """
    available_cities =  os.listdir(location)
    matches = [match for match in available_cities if city in match]
    if len(matches) == 0:
        raise Exception("city not found")
    elif len(matches) > 1:
        raise Exception("city ambiguous")
    else:
        data = load_list(matches[0],location=location)

    datetime64s = np.array(data[3]["date"])
    index = min(range(len(datetime64s)),
                key=lambda i: abs(datetime64s[i]-np.datetime64(datetime)))

    if datetime64s[index]-np.datetime64(datetime) > np.timedelta64(13,'h'):
        raise Exception(f"date {datetime} not in the database")
    else:
        pass

    state = [data[3][att][index] for att in attributes]

    return state




if __name__ == "__main__":
    print_map(get_unique_cities())


















