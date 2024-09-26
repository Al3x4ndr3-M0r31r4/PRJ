import matplotlib.pyplot as plt
import numpy as np


class DatasetAnalysis:

    def class_dist(self, y, title=None, labels=None, colors=None, title_font_size=None, label_font_size=None):
        classes = np.unique(y, return_counts=True)

        title_fs = 18
        if title_font_size is not None:
            title_fs = title_font_size

        if title is not None:
            plt.title(title, fontsize=title_fs, pad=20)

        label_fs = 10
        if label_font_size is not None:
            label_fs = label_font_size

        labels_plt = classes[0]
        if labels is not None:
            labels_plt = labels

        if colors is not None:
            plt.bar(labels_plt, classes[1], color=colors)
        else:
            plt.bar(labels_plt, classes[1])


        classes_perc = [f"{x} ({round(100 * x / len(y), 1)}%)" for x in classes[1]]

        ax = plt.gca()
        ax.bar_label(ax.containers[0], labels=classes_perc, label_type='edge', fontsize=label_fs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=label_fs)
        ax.tick_params(axis="x", labelsize=label_fs)


    def feat_hist(self, X, n_rows, n_cols, feat_names=None, bins=None, figsize=(10, 10), color=None, title_font_size=None, label_font_size=None):
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        ax = ax.flatten()

        feature_names = feat_names

        n_bins = bins

        if n_bins is None:
            n_bins = "auto"

        if feature_names is None:
            feature_names = range(X.shape[1])


        for i in range(X.shape[1]):
            ax[i].hist(X[:, i], bins=n_bins, color=color)
            ax[i].set_title(f"{feature_names[i]}", fontsize=title_font_size)
            ax[i].tick_params(axis='both', which='major', labelsize=label_font_size)

        for i in range(X.shape[1], len(ax)):
            ax[i].axis("off")

