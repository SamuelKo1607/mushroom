import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn import metrics
from sklearn import model_selection

from gather_data import load_all_posts
from fitting import hand_jar
from fitting import z_norm







#%%
if __name__ == "__main__":
    df = hand_jar(load_all_posts(rated_only=True))

    df_pos = df[df["rating"] > 0]
    df_neg = df[df["rating"] == 0]

    df = pd.concat([df_pos,
                    df_neg[np.random.choice([True,False],
                                            p=[0.1,0.9],
                                            size=len(df_neg))]
                    ])

    y = (df["rating"]>0).astype(int)
    x = df.drop(columns=['rating'])

    for col in x.columns:
        x[col] = z_norm(x[col])

    (x_train, x_test,
     y_train, y_test) = model_selection.train_test_split(x, y, test_size=.1)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=128, activation='sigmoid',
                    input_dim=len(x_train.columns)))
    model.add(keras.layers.Dense(units=8, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd',
                  metrics='accuracy')

    model.fit(x_train, y_train, epochs=1000, batch_size=128)

    y_hat = model.predict(x_test)
    print(metrics.accuracy_score([0 if val < 0.5 else 1 for val in y_test],
                                 [0 if val < 0.5 else 1 for val in y_hat]))
    plt.violinplot([y_hat[y_test==0,0],y_hat[y_test==1,0]],[0,1])
    plt.ylim(0,1)
    plt.show()








