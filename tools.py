import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot

d = {
    'Alder':{
        '18-24':0.12,
        '25-34':0.16,
        '35-44':0.19,
        '45-54':0.19,
        '55-64':0.17,
        '65+':0.09,
        '18-':0.08,
    },
    'Køn':{
        'M':0.51,
        'K':0.49
    },
    'Region':{
        'Hovedstaden':0.30,
        'Sjælland':0.14,
        'Syddanmark':0.21,
        'Midtjylland':0.22,
        'Nordjylland':0.11,
        'Ved ikke':0.02,
    }
}

def get_sample_ind(N=1000, dists=None):
    all_sample = []
    for i in range(N):
        sample = []
        for dist in dists:
            groups = list(dist.keys())
            props = [dist[j] for j in groups]
            #print(groups, np.sum(props))
            s = np.random.choice(groups, 1, p=props)
            sample.append(s[0])
        #print(sample)
        all_sample.append(sample)
    return all_sample

def get_sample(N=1000, dists=None):
    sample = []
    for dist in dists:
        groups = list(dist.keys())
        props = [dist[j] for j in groups]
        #print(groups, np.sum(props))
        s = np.random.choice(groups, N, p=props)
        sample.append(list(s))
    # Zip together
    all_sample = list(zip(*sample))
    return all_sample

def get_std_sample(N=1000):
    # Create dataframe
    labels = ['Alder', 'Køn', 'Region']

    # Define a distribution
    dists = [d[labels[0]], d[labels[1]], d[labels[2]]]
    sample = get_sample(N=N, dists=dists)
    # Make data frame
    df = pd.DataFrame.from_records(sample, columns=labels)
    df['Alle'] = pd.Series(['Alle']*N, index=df.index)
    df['val'] = pd.Series(np.ones(N), index=df.index)

    # Create info
    df_i = {}
    for label in labels:
        # Count by labels
        df_s = df.groupby(label)[label].count() / df[label].count() * 100
        # Create a pandas Series
        d_s = pd.Series(d[label]) * 100
        d_s.name = label+"_DST"
        # Concat
        df_c = pd.concat([df_s, d_s], axis=1)
        df_i[label] = df_c

    return df, df_i
    
def create_rand_series(df=None, x_avg=None, ex=None):
    N = len(df.index)
    # From the fractional / relative uncertainty, calculate the standard deviation
    sigma = ex * x_avg
    rand_nrs = sigma * np.random.randn(N) + x_avg
    # Create Series
    s = pd.Series(rand_nrs, index=df.index)
    return s
    
def create_boxplot_hist(df=None, col=None):
    # Plot histogram af punkters
    f, (ax1, ax2) = plt.subplots(2,1, figsize=(10,5), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    f.subplots_adjust(hspace=0)

    v = df[col]
    ax1.boxplot(v, notch=False, sym=None, vert=False)

    mean = np.mean(v)
    sigma = np.std(v)
    x = np.linspace(min(v), max(v), len(v))
    # Plot
    ax2.hist(v, normed=True, bins=20)
    p = ax2.plot(x, mlab.normpdf(x, mean, sigma))
    c = p[-1].get_color() 
    ax2.axvline(x=180, c="k")
    ax2.axvline(x=mean, c=c)
    ax2.axvline(x=mean+sigma, c="b")
    ax2.axvline(x=mean-sigma, c="b")
    ax2.axvline(x=mean+2*sigma, c="b")
    ax2.axvline(x=mean-2*sigma, c="b")
    return f, ax1, ax2, mean, sigma