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


def tsne(df):
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
    y = df["rating"]>0

    # fitting
    tsne = TSNE(n_components=2, verbose=1, random_state=124)
    z = tsne.fit_transform(x) 
    result_df = pd.DataFrame()
    result_df["y"] = y
    result_df["comp-1"] = z[:,0]
    result_df["comp-2"] = z[:,1]

    mask = df["rating"]>-1

    # plot
    fig,ax = plt.subplots()
    scat = ax.scatter(result_df["comp-1"][mask],result_df["comp-2"][mask],
                      c=result_df["y"][mask],
                      s=5,alpha=0.5,cmap="jet",lw=0)
    cb = fig.colorbar(scat, label = "rating")
    ax.set_xlabel("comp-1")
    ax.set_ylabel("comp-2")
    fig.show()

    return result_df


def logit_probability(coeffs,covariates=None):
    """
    Parameters
    ----------
    coeffs : np.array 1D of float
        The fitted values.
    covariates : np.array 1D of float, optional
        The observed parameters. The default is None.

    Returns
    -------
    probty : float
        The probability.

    """
    if covariates is None:
        covariates = np.ones(len(coeffs))

    prediction = np.sum(coeffs*covariates)
    probty = 1 / (1 + np.exp(-prediction))
    return probty


def log_prior(coeffs):
    """
    The most vanilla prior ever.

    Parameters
    ----------
    coeffs : np.array 1D of float
        the value of parameters.

    Returns
    -------
    logprior : float
        Logarithmic prior.

    """
    logprior = (-np.sum(coeffs**2))
    return logprior


def log_likelihood(x,y,coeffs):
    """
    The log likelihood of coeffs, given covariates x and data y.

    Parameters
    ----------
    x : np.array 2D of shape (n,m)
        DESCRIPTION.
    y : np.array 1D of shape (n)
        DESCRIPTION.
    coeffs : np.array 1D of shape (m+1)
        DESCRIPTION.

    Raises
    ------
    Exception
        if the sizes don't match.

    Returns
    -------
    float
        the sum loglik.

    """
    if x.shape[1]+1 != len(coeffs):
        raise Exception("x.shape[1]+1 != len(coeffs)")

    coeffs_tiled = np.tile(coeffs,(x.shape[0],1)).astype(np.longdouble)
    x_tiled = np.vstack((np.ones(x.shape[0]),
                         x.transpose())).transpose().astype(np.longdouble)

    logodds = np.sum(coeffs_tiled*x_tiled,axis=1)
    p = 1 / (1 + np.exp(-logodds))

    logliks = y*np.log(p) + (1-y)*np.log(1-p)

    return np.sum(logliks)


def proposal(theta,
             scale=1e-3,
             family="uniform"):
    """
    Generates a new theta (proposal value for MH) and checks that it is 
    actually within allowed values (prior).

    Parameters
    ----------
    theta : list
        Old coefficients list.
    scale : float, optional
        The scale of change, smaller means smaller deviation of the proposal.
    family : str, optional
        The family of the proposal. One of "normal", "uniform". 
        The default is "uniform."

    Raises
    ------
    Exception
        if the requested family is not defined.

    Returns
    -------
    proposed_theta : TYPE
        Proposed theta (allowed by the prior).

    """

    if family == "normal":
        rand = np.random.normal(0,1,size=len(theta))*scale
    elif family == "uniform":
        rand = np.random.uniform(-1,1,size=len(theta))*scale
    else:
        raise Exception(f"unknonwn family: {family}")

    proposed_theta = theta+rand

    return proposed_theta


def step(theta,
         x,
         y,
         **kwargs):
    """
    Performs a step, returns either the old or the new theta.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : 
    scale : float, optional
        The scale of the proposed change.
    family : str, optional
        The family of the proposal. One of "normal", "uniform". 
        The default is "uniform."


    Returns
    -------
    theta : TYPE
        DESCRIPTION.
    change : bool
        Whether the new theta is actually new, or just the old one.

    """
    old_goodness = log_likelihood(x,y,theta) + log_prior(theta)

    proposed_theta = proposal(theta,**kwargs)
    proposed_goodness = (log_likelihood(x,y,proposed_theta)
                         + log_prior(proposed_theta))

    log_acc_ratio = proposed_goodness - old_goodness
    threshold = np.random.uniform(0,1)

    if np.exp(log_acc_ratio) > threshold:
        change = True
        theta = proposed_theta
    else:
        change = False

    return theta, change


def fit_logit_bayesian(df,
                       burnin=1000,
                       samples=10000,
                       coeffs0=np.array([-0.05,
                                         -0.03,
                                         -0.01,
                                         -0.09,
                                         0.04,
                                         -0.01,
                                         0.05,
                                         -0.1,
                                         0.03,
                                         -0.05,
                                         -0.09,
                                         -0.1])):
    """
    GLM bayesian.

    Parameters
    ----------
    df : pd.dataframe
        as from hand_jar.
    burnin : int, optional
        DESCRIPTION. The default is 1000.
    samples : int, optional
        DESCRIPTION. The default is 10000.
    coeffs0 : np.array of float, optional
        Start coefficients. The default is np.array([-0.05,                                         -0.03,                                         -0.01,                                         -0.09,                                         0.04,                                         -0.01,                                         0.05,                                         -0.1,                                         0.03,                                         -0.05,                                         -0.09,                                         -0.1]).

    Returns
    -------
    sampled : np.array of float
        Samples, shape (burnin+samples, 12).

    """

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
    y = df["rating"]>0

    theta = coeffs0
    changes = 0
    sampled = np.zeros(shape=(0,12))
    sampled = np.vstack((sampled,np.array(theta)))

    total=burnin+samples
    for i in tqdm(range(total)):
        theta, change = step(theta,x,y)
        if change:
            changes += change
        sampled = np.vstack((sampled,np.array(theta)))

    print(f"Acc. rate = {changes/total}")
    return sampled


def z_norm(array):
    return (array-np.mean(array))/np.std(array)


def backprop_nn(df,
                iterations=1000,
                hidden_neurons=7,
                steepness=1,
                conv_speed=0.1):

    a = np.vstack((z_norm(df["week"]),
                   z_norm(df["rainfall_3d"]),
                   z_norm(df["rainfall_7d"]),
                   z_norm(df["precip_hours_3d"]),
                   z_norm(df["precip_hours_7d"]),
                   z_norm(df["temp_3d"]),
                   z_norm(df["temp_7d"]),
                   z_norm(df["mintemp_3d"]),
                   z_norm(df["mintemp_7d"]),
                   z_norm(df["evapotranspiration_3d"]),
                   z_norm(df["evapotranspiration_7d"])
                   )).transpose()
    b = df["rating"]/5
    
    def s(x,l=1): # l = koeficient rýchlosti
        return 1/(1+np.exp(-x*l))
    
    def ds(x,l=1):
        return l*s(x,l)*(1-s(x,l))
    
    def xor(x,y):
        return (int(round(x))^int(round(y))
                )+np.random.normal(0,0.005) #blízke 0 a 1, ale nie ostré

    w = (np.random.random((hidden_neurons,
                           a.shape[1]+1))-0.5)/10 # koeficienty v prvej vrstve
    dw = np.zeros((hidden_neurons,
                   a.shape[1]+1)) # koeficienty v prvej vrstve - delta

    q = (np.random.random(hidden_neurons+1)-0.5)/10 # koeficienty v druhej vrstve
    dq = np.zeros(hidden_neurons+1) # koeficienty v druhej vrstve - delta

    m=np.zeros(hidden_neurons) # výstupy prvej vstrvy
    
    def val(m,q):
        s_val = -q[0] + np.sum(m*q[1:])
        return s_val
    
    def E(a, b, w, q, m): # výpočet chybovej funkcie
        hits = 0
        for i in range(len(b)): # tréningové vstupy
            m = s(( -1*(w[:,0]) + np.sum(w[:,1:]*a[i,:],axis=1))
                  ,steepness) # skrytá vrstva
            hits += (np.round(s((val(m,q)),steepness))
                     == np.round(b[i]))
        return hits

    def delta_out(a2,b2,w,q,m):  #a2 = 1D vektor dĺžky 11, b2 = číslo
        m = s(( -1*(w[:,0]) + np.sum(w[:,1:]*a2,axis=1)) ,steepness)
        sigma_l = val(m,q)    
        pref = (b2-s((sigma_l),steepness))*ds((sigma_l),steepness)
        dq = np.append(-conv_speed*pref,
                       conv_speed*pref*m)   #vektor dĺžky N1+1 = vektor dq
        return dq
    
    def delta_hidden(a2,b2,w,q,m):  #a2 = 1D vektor dĺžky 11, b2 = číslo
        m = s(( -1*(w[:,0]) + np.sum(w[:,1:]*a2,axis=1)) ,steepness)
        sigma_l = val(m,q)
        pref = (b2-s((sigma_l),steepness))*ds((sigma_l),steepness)
        for i in range(hidden_neurons): #cez všetky skryté neuróny
            sigma_i = -w[i,0] + np.sum(a2*w[i,1:])
            dw[i,0]= pref*q[i+1]*ds((sigma_i),
                                    steepness)  # -*-: jedno z triviálneho vstupu do skrytej vrstvy a jedno z derivácie
            for j in range(len(a2)):   #cez všetky vstupy
                dw[i,j+1]= -pref*q[i+1]*ds((sigma_i),steepness)*a2[j]
        return dw

    def learn(rounds, a, b, w, q, m, dw, dq):
        for i in tqdm(range(rounds)):
            if np.random.random() >= 0.5:
                pick = np.random.choice(np.arange(len(b))[b>0.5])
            else:
                pick = np.random.choice(np.arange(len(b))[b<0.5])
            dq = delta_out(a[pick,:],b[pick],w,q,m)
            dw = delta_hidden(a[pick,:],b[pick],w,q,m)
            q += dq
            w += dw
        
    print(E(a, b, w, q, m))
    learn(iterations, a, b, w, q, m, dw, dq)
    print(E(a, b, w, q, m))
    return w,q


#%%
if __name__ == "__main__":
    df = hand_jar(load_all_posts(rated_only=True))

    # t-SNE
    result_df = tsne(df)


    # logit Bayesian
    sample = fit_logit_bayesian(df)


    w,q = backprop_nn(df,100000)
    plt.imshow(w)
    plt.show()
    plt.plot(q)
    plt.show()
    

    
    
    
    
    
