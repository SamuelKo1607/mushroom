import os
import pandas as pd
import numpy as np
from time import sleep
import datetime as dt
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.manifold import TSNE

from gather_data import load_all_posts
from gather_weather import load_list

def create_df(posts):
    """
    Creates a dataframe of mushroom reports 
    with weather and calendar features.

    Parameters
    ----------
    posts : list of object of class Post.
        All the posts data as per gather_data.load_all_posts.

    Returns
    -------
    all_cities : pd.df
        All the data containing calendar and weather info.

    """
    rated_posts = pd.DataFrame([post.__dict__ for post in posts])
    gb = rated_posts.groupby('city')
    list_of_df_by_city = [gb.get_group(x).copy() for x in gb.groups]
    for num,df in enumerate(list_of_df_by_city):
        print()
        print(df.iloc[0].city+f", {num}/{len(list_of_df_by_city)}")
        # Calendar features
        for i in df.index:
            calendar = df.loc[i].datetime.isocalendar()
            df.loc[i, "year"] = calendar[0]
            df.loc[i, "week"] = calendar[1]
            df.loc[i, "weekday"] = calendar[2]
            df.loc[i, "city_num"] = num
        # Weather features
        weather = load_list(df.iloc[0].city+".pkl",
                            location = os.path.join("997_weather_data",""))[3]
        for j in tqdm(df.index, leave=True):
            w_3d_mask = ((weather.date<df.datetime[j])
                         *(weather.date>=df.datetime[j]
                           -dt.timedelta(3)))
            w_7d_mask = ((weather.date<df.datetime[j])
                         *(weather.date>=df.datetime[j]
                           -dt.timedelta(7)))
            df.loc[j, "rainfall_3d"] = np.mean(weather[w_3d_mask].
                                               rain_sum_mm)
            df.loc[j, "rainfall_7d"] = np.mean(weather[w_7d_mask].
                                               rain_sum_mm)
            df.loc[j, "precip_hours_3d"] = np.mean(weather[w_3d_mask].
                                                   precipitation_hours_h)
            df.loc[j, "precip_hours_7d"] = np.mean(weather[w_7d_mask].
                                                   precipitation_hours_h)
            df.loc[j, "temp_3d"] = np.mean(weather[w_3d_mask].
                                           temperature_2m_mean_C)
            df.loc[j, "temp_7d"] = np.mean(weather[w_7d_mask].
                                           temperature_2m_mean_C)
            df.loc[j, "mintemp_3d"] = np.mean(weather[w_3d_mask].
                                              temperature_2m_min_C)
            df.loc[j, "mintemp_7d"] = np.mean(weather[w_7d_mask].
                                              temperature_2m_min_C)
            df.loc[j, "evapotranspiration_3d"] = np.mean(weather[w_3d_mask].
                                                         evapotranspiration)
            df.loc[j, "evapotranspiration_7d"] = np.mean(weather[w_7d_mask].
                                                         evapotranspiration)
    # Union all the city-specific dataframes
    all_cities = pd.concat(list_of_df_by_city)
    return all_cities


def balanced_df(df,column="is_positive"):
    """
    Balances the data set by a selected boolean columns.

    Parameters
    ----------
    df : pd.df
        The dataframe to balance.
    column : str, optional
        Column key of the boolean column to balance based on. 
        The default is "is_positive".

    Returns
    -------
    balanced : pd.df
        The randomly balanced df.

    """
    df_pos = df[df[column]].copy()
    df_neg = df[~df[column]].copy()

    df_pos_sample = df_pos.sample(min(len(df_pos.index),
                                      len(df_neg.index))).copy()
    df_neg_sample = df_neg.sample(min(len(df_pos.index),
                                      len(df_neg.index))).copy()
    balanced = pd.concat([df_pos_sample,df_neg_sample])
    return balanced


#%%
if __name__ == "__main__":
    try:
        balanced = pd.read_csv(os.path.join("995_pickled","data_balanced.csv"))
    except FileNotFoundError:
        try:
            df = pd.read_csv(os.path.join("995_pickled","data_complete.csv"))
        except FileNotFoundError:
            posts = load_all_posts(rated_only=False)
            df = create_df(posts)
            df.to_csv(os.path.join("995_pickled","data_complete.csv"),
                      index=False)
        finally:
            balanced = balanced_df(df)
            balanced.to_csv(os.path.join("995_pickled","data_balanced.csv"),
                            index=False)
    finally:
        features = ['city_num', 'year', 'week', 'weekday', 'rainfall_3d',
                    'rainfall_7d', 'precip_hours_3d', 'precip_hours_7d',
                    'temp_3d', 'temp_7d', 'mintemp_3d', 'mintemp_7d',
                    'evapotranspiration_3d', 'evapotranspiration_7d']
        x = balanced[features[4:]]
        y = balanced['is_positive']

#%%
    # K-NN
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"K-NN accuracy: {accuracy}")
    #print(classification_report(y_test, y_pred))

    # GradientBoostingClassifier
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    gb_clf = GradientBoostingClassifier(n_estimators=500,
                                        learning_rate=0.1)
    gb_clf.fit(x_train, y_train)
    y_pred = gb_clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient boosting accuracy: {accuracy}")
    #print(classification_report(y_test, y_pred))

    # LogisticRegression
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(x_train, y_train)
    y_pred = log_reg.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Log. regression accuracy: {accuracy}")
    #print(classification_report(y_test, y_pred))

    # RandomForestClassifier
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    random_forest = RandomForestClassifier(n_estimators=1_000)
    random_forest.fit(x_train, y_train)
    y_pred = random_forest.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random forest accuracy: {accuracy}")
    #print(classification_report(y_test, y_pred))

    # TSNE
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    y_train.index=np.arange(len(y_train.index))
    y_test.index=np.arange(len(y_test.index))+len(y_train.index)
    tsne = TSNE(n_components=3)
    z = tsne.fit_transform(pd.concat([x_train, x_test]))
    df_z = pd.DataFrame(z)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(df_z[:len(x_train.index)], y_train)
    y_pred = knn.predict(df_z[-len(x_test.index):])
    accuracy = accuracy_score(y_test, y_pred)
    print(f"TSNE accuracy: {accuracy}")
    #print(classification_report(y_test, y_pred))




