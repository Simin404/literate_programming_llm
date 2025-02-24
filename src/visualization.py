from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.impute import SimpleImputer
import colorcet as cc
import time
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from matplotlib.cm import get_cmap
from sklearn.manifold import TSNE
import plotly.express as px
import torch


def remove_nan_rows(tensor):
    # Find rows with any NaN values
    nan_mask = torch.isnan(tensor).any(dim=1)
    
    # Indices of rows to remove
    removed_indices = torch.where(nan_mask)[0].tolist()
    
    # Tensor without rows containing NaN
    cleaned_tensor = tensor[~nan_mask]
    
    return cleaned_tensor, removed_indices

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
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    return df


def vis_2d(emb, part_data, model_name, data_type = 'language'):
    if model_name == 'embedding_ada':
        emb, removed = remove_nan_rows(emb)
        part_data = part_data.drop(index=removed).reset_index(drop=True)
    df = tsne(emb, part_data)
    total_num = len(set(part_data['language']))
    palette = sns.color_palette(cc.glasbey_light, n_colors=total_num)
    plt.figure(figsize=(10,8))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=part_data[data_type],
        style=part_data[data_type],
        palette=palette,
        data=df,
        legend='full',
        alpha= 1
    )
    plt.savefig('image/tsne_2d/' + model_name +'.png')

def set_color(num, languages, labels):
    cmap = get_cmap('tab10', num)  # Use tab10 for up to 10 colors (adjust for larger)
    colors = [cmap(i) for i in range(10)]

    color_map={}
    for i in range(10):
        color_map[languages[i]]=colors[i]
    color_lables=[]
    for label in labels:
        color_lables.append(color_map[label])
    return color_lables


def vis_3d(emb, languages, labels, model_name):
    if model_name == 'embedding_ada':
        emb, removed = remove_nan_rows(emb)
        for index in sorted(removed, reverse=True):
            del labels[index]
    tsne = TSNE(n_components=3, random_state=0)
    data_3d = tsne.fit_transform(emb)
    color_lables = set_color(10, languages, labels)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], color=color_lables,label=labels, marker='o',alpha=0.8, s=50)
    # Labels and title
    ax.set_title(model_name)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    fig.savefig('image/tsne_3d/' + model_name + '.png')
    # plt.show()