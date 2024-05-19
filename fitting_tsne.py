import numpy as np
import datetime as dt
import pandas as pd
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

from gather_data import load_all_posts
from gather_weather import load_weather


rated_posts = load_all_posts(rated_only=True)

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

for i in tqdm(range(len(rated_posts))):

    post = rated_posts[i]
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

    rainfall_3d = np.append(rainfall_3d,np.sum(w[:3,0]))
    rainfall_7d = np.append(rainfall_7d,np.sum(w[:7,0]))
    precip_hours_3d = np.append(precip_hours_3d,np.sum(w[:3,1]))
    precip_hours_7d = np.append(precip_hours_7d,np.sum(w[:7,1]))
    temp_3d = np.append(temp_3d,np.mean(w[:3,2]))
    temp_7d = np.append(temp_7d,np.mean(w[:7,2]))
    mintemp_3d = np.append(mintemp_3d,np.mean(w[:3,3]))
    mintemp_7d = np.append(mintemp_7d,np.mean(w[:7,3]))
    evapotranspiration_3d = np.append(evapotranspiration_3d,np.mean(w[:3,4]))
    evapotranspiration_3d = np.append(evapotranspiration_3d,np.mean(w[:7,4]))


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

df.to_pickle("datacube.pkl")

print(df)

"""

x = np.vstack((features["left_right"][mask],
               features["bottom_top"][mask],
               #features["polarities"][mask],
               np.log10(features["stupid_max"][mask][:,0]),
               np.log10(features["stupid_max"][mask][:,1]),
               np.log10(features["stupid_max"][mask][:,2]),
               np.log10(features["delays"][mask][:,0]),
               np.log10(features["delays"][mask][:,1]),
               np.log10(features["delays"][mask][:,2]),
               np.log10(features["electron_amplitudes"][mask]),
               np.log10(features["body_risetimes"][mask]),
               features["heliocentric_distances"][mask],
               features["azimuthal_velocities"][mask],
               features["radial_velocities"][mask])).transpose()

#normalization
for feature in range(len(x[0,:])):
    x[:,feature]-=np.mean(x[:,feature])
    x[:,feature]/=np.std(x[:,feature])


"""