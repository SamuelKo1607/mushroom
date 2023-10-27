import numpy as np
import datetime as dt
from sklearn.manifold import TSNE


from gather_data import load_all_posts
from gather_weather import load_weather






rated_posts = load_all_posts(rated_only=True)

rating = []
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

for i, post in enumerate(rated_posts):

    print(i)

    rating.append(post.rate)

    rainfall_3d.append(sum([load_weather(post.city,
                                         post.datetime - dt.timedelta(delta),
                                         attributes=['rain_sum_mm'])[0]
                            for delta in range(1,4)]))

    rainfall_7d.append(sum([load_weather(post.city,
                                         post.datetime - dt.timedelta(delta),
                                         attributes=['rain_sum_mm'])[0]
                            for delta in range(4,8)]))

    precip_hours_3d.append(sum([load_weather(post.city,
                                             post.datetime - dt.timedelta(delta),
                                             attributes=['precipitation_hours_h'])[0]
                                for delta in range(1,4)]))

    precip_hours_7d.append(sum([load_weather(post.city,
                                             post.datetime - dt.timedelta(delta),
                                             attributes=['precipitation_hours_h'])[0]
                                for delta in range(4,8)]))

    temp_3d.append(np.mean([load_weather(post.city,
                                         post.datetime - dt.timedelta(delta),
                                         attributes=['temperature_2m_mean_C'])[0]
                            for delta in range(1,4)]))

    temp_7d.append(np.mean([load_weather(post.city,
                                         post.datetime - dt.timedelta(delta),
                                         attributes=['temperature_2m_mean_C'])[0]
                            for delta in range(4,8)]))

    mintemp_3d.append(np.mean([load_weather(post.city,
                                            post.datetime - dt.timedelta(delta),
                                            attributes=['temperature_2m_min_C'])[0]
                               for delta in range(1,4)]))

    mintemp_7d.append(np.mean([load_weather(post.city,
                                            post.datetime - dt.timedelta(delta),
                                            attributes=['temperature_2m_min_C'])[0]
                               for delta in range(4,8)]))

    evapotranspiration_3d.append(np.mean([load_weather(post.city,
                                                       post.datetime - dt.timedelta(delta),
                                                       attributes=['evapotranspiration'])[0]
                                          for delta in range(1,4)]))

    evapotranspiration_7d.append(np.mean([load_weather(post.city,
                                                       post.datetime - dt.timedelta(delta),
                                                       attributes=['evapotranspiration'])[0]
                                          for delta in range(4,8)]))


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