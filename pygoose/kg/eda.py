import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_real_feature(df, feature_name, bins=50, figsize=(15, 15)):
    """
    Plot the distribution of a real-valued feature conditioned by the target.

    Examples:
        `plot_real_feature(X, 'emb_mean_euclidean')`

    Args:
        df: Pandas dataframe containing the target column (named 'target').
        feature_name: The name of the feature to plot.
        bins: The number of histogram bins for the distribution plot.
        figsize: The size of the plotted figure.
    """

    ix_negative_target = df[df.target == 0].index
    ix_positive_target = df[df.target == 1].index

    plt.figure(figsize=figsize)

    ax_overall_dist = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_target_conditional_dist = plt.subplot2grid((3, 2), (1, 0), colspan=2)

    ax_botplot = plt.subplot2grid((3, 2), (2, 0))
    ax_violin_plot = plt.subplot2grid((3, 2), (2, 1))

    ax_overall_dist.set_title('Distribution of {}'.format(feature_name), fontsize=16)
    sns.distplot(
        df[feature_name],
        bins=50,
        ax=ax_overall_dist
    )

    sns.distplot(
        df.loc[ix_positive_target][feature_name],
        bins=bins,
        ax=ax_target_conditional_dist,
        label='Positive Target'
    )
    sns.distplot(
        df.loc[ix_negative_target][feature_name],
        bins=bins,
        ax=ax_target_conditional_dist,
        label='Negative Target'
    )
    ax_target_conditional_dist.legend(loc='upper right', prop={'size': 14})

    sns.boxplot(
        y=feature_name,
        x='target',
        data=df,
        ax=ax_botplot
    )
    sns.violinplot(
        y=feature_name,
        x='target',
        data=df,
        ax=ax_violin_plot
    )

    plt.show()


def plot_pair(df, feature_name_1, feature_name_2, kind='scatter', alpha=0.01, **kwargs):
    """
    Plot a scatterplot of two features against one another,
    and calculate Pearson correlation coefficient.

    Examples:
        `plot_pair(X, 'emb_mean_euclidean', 'emb_mean_cosine')`

    Args:
        df:
        feature_name_1: The name of the first feature.
        feature_name_2: The name of the second feature.
        kind: One of the values { 'scatter' | 'reg' | 'resid' | 'kde' | 'hex' }.
        alpha: Alpha channel value.
        **kwargs: Additional argument to 'sns.jointplot'.
    """

    plt.figure()
    sns.jointplot(
        feature_name_1,
        feature_name_2,
        df,
        alpha=alpha,
        kind=kind,
        **kwargs
    )
    plt.show()


def plot_feature_correlation_heatmap(df, features, font_size=9, figsize=(15, 15), save_filename=None):
    """
    Plot a correlation heatmap between every feature pair.

    Args:
        df: Pandas dataframe containing the target column (named 'target').
        features: The list of features to include in the correlation plot.
        font_size: Font size for heatmap cells and axis labels.
        figsize: The size of the plot.
        save_filename: (Optional) The path of the file to save a high-res version of the plot to.
    """

    features = features[:]
    features += ['target']

    mcorr = df[features].corr()
    mask = np.zeros_like(mcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(
        mcorr,
        mask=mask,
        cmap=cmap,
        square=True,
        annot=True,
        fmt='0.2f',
        annot_kws={'size': font_size},
    )

    heatmap.tick_params(axis='both', which='major', labelsize=font_size)
    heatmap.tick_params(axis='both', which='minor', labelsize=font_size)

    heatmap.set_xticklabels(features, rotation=90)
    heatmap.set_yticklabels(reversed(features))

    plt.show()

    if save_filename is not None:
        fig.savefig(save_filename, dpi=300)


def scatterplot_matrix(df, features, downsample_frac=None, figsize=(15, 15)):
    """
    Plot a scatterplot matrix for a list of features, colored by target value.

    Example: `scatterplot_matrix(X, X.columns.tolist(), downsample_frac=0.01)`

    Args:
        df: Pandas dataframe containing the target column (named 'target').
        features: The list of features to include in the correlation plot.
        downsample_frac: Dataframe downsampling rate (0.1 to include 10% of the dataset).
        figsize: The size of the plot.
    """

    if downsample_frac:
        df = df.sample(frac=downsample_frac)

    plt.figure(figsize=figsize)
    sns.pairplot(df[features], hue='target')
    plt.show()
