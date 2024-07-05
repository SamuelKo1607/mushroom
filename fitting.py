import os
import numpy as np
import datetime as dt
import pandas as pd
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

from gather_data import load_all_posts
from gather_weather import load_weather


def sterilize(posts,
              path):
    """
    shape a data cube and pickle

    Parameters
    ----------
    posts : list
        ist of Post object.
    path : str
        The path to the pickle.

    Returns
    -------
    df : pd.datadaframe
        The data as shaped an pickled.

    """
    rating = []
    week = []
    rainfall_3d = []
    rainfall_7d = []
    precip_hours_3d = []
    precip_hours_7d = []
    temp_3d = []
    temp_7d = []
    mintemp_3d = []
    mintemp_7d = []
    evapotranspiration_3d = []
    evapotranspiration_7d = []
    
    for i in tqdm(range(len(posts))):
    
        post = posts[i]
        rating.append(post.rate)
        week.append(post.datetime.isocalendar()[1])
    
        w = np.zeros((0,5))
        for delta in range(1,8):
            w = np.vstack((w,load_weather(post.city,
                                          post.datetime - dt.timedelta(delta),
                                          attributes=['rain_sum_mm',
                                                      'precipitation_hours_h',
                                                      'temperature_2m_mean_C',
                                                      'temperature_2m_min_C',
                                                      'evapotranspiration'])))
    
        rainfall_3d = np.append(rainfall_3d,np.mean(w[:3,0]))
        rainfall_7d = np.append(rainfall_7d,np.mean(w[:7,0]))
        precip_hours_3d = np.append(precip_hours_3d,np.mean(w[:3,1]))
        precip_hours_7d = np.append(precip_hours_7d,np.mean(w[:7,1]))
        temp_3d = np.append(temp_3d,np.mean(w[:3,2]))
        temp_7d = np.append(temp_7d,np.mean(w[:7,2]))
        mintemp_3d = np.append(mintemp_3d,np.mean(w[:3,3]))
        mintemp_7d = np.append(mintemp_7d,np.mean(w[:7,3]))
        evapotranspiration_3d = np.append(evapotranspiration_3d,np.mean(w[:3,4]))
        evapotranspiration_7d = np.append(evapotranspiration_7d,np.mean(w[:7,4]))

    df = pd.DataFrame({'rating': rating,
                       'week': week,
                       'rainfall_3d': rainfall_3d,
                       'rainfall_7d': rainfall_7d,
                       'precip_hours_3d': precip_hours_3d,
                       'precip_hours_7d': precip_hours_7d,
                       'temp_3d': temp_3d,
                       'temp_7d': temp_7d,
                       'mintemp_3d': mintemp_3d,
                       'mintemp_7d': mintemp_7d,
                       'evapotranspiration_3d': evapotranspiration_3d,
                       'evapotranspiration_7d': evapotranspiration_7d})
    
    df.to_pickle(path)

    return df


def hand_jar(posts,
             path=os.path.join("995_pickled","datacube.pkl")):
    """
    Either loads or creates, saves and loads the data.

    Parameters
    ----------
    posts : list
        ist of Post object.
    path : str, optional
        The path to the pickle. 
        The default is os.path.join("995_pickled","datacube.pkl").

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    try:
        df = pd.read_pickle(path)
    except:
        df = sterilize(posts,path)

    return df




#%% t-SNE

if __name__ == "__main__":
    df = hand_jar(load_all_posts(rated_only=True))
    print(df)

    # data
    x = np.vstack((df["week"],
                   df["rainfall_3d"],
                   df["rainfall_7d"],
                   df["precip_hours_3d"],
                   df["precip_hours_7d"],
                   df["temp_3d"],
                   df["temp_7d"],
                   df["mintemp_3d"],
                   df["mintemp_7d"],
                   df["evapotranspiration_3d"],
                   df["evapotranspiration_7d"]
                   )).transpose()
    y = df["rating"]

    # fitting
    tsne = TSNE(n_components=2, verbose=1, random_state=124)
    z = tsne.fit_transform(x) 
    result_df = pd.DataFrame()
    result_df["y"] = y
    result_df["comp-1"] = z[:,0]
    result_df["comp-2"] = z[:,1]

    mask = df["rating"]>1

    # plot
    fig,ax = plt.subplots()
    scat = ax.scatter(result_df["comp-1"][mask],result_df["comp-2"][mask],
                      c=result_df["y"][mask],
                      s=5,alpha=0.5,cmap="jet",lw=0)
    cb = fig.colorbar(scat, label = "rating")
    ax.set_xlabel("comp-1")
    ax.set_ylabel("comp-2")
    fig.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
