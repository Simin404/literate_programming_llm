from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.impute import SimpleImputer
import colorcet as cc
import torch
import numpy as np
import time
import pandas as pd


def tsne(embeddings, part_data):

    print(embeddings.shape, len(set(part_data['language'])), len(set(part_data['task'])))
    feat_cols = ['embedding'+str(i) for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings.numpy(),columns=feat_cols)
    imp = SimpleImputer(strategy="mean")
    imp.fit(df)
    df = imp.transform(df)
    df = pd.DataFrame(df)
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    return df



def visualize(df, part_data, model, data_type = 'language'):
    total_num = len(set(part_data['language']))
    palette = sns.color_palette(cc.glasbey_light, n_colors=total_num)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=part_data[data_type],
        style=part_data[data_type],
        palette=palette,
        data=df,
        legend='full',
        alpha= 1
    )
    plt.savefig('image/' + model + '_' +str(total_num) + '_languages.png')