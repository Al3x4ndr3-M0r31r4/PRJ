import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import shap
from PerformanceAnalysis import PerformanceAnalysis
class Visualization:

    def __init__(self):
        self.perf_analysis = PerformanceAnalysis()
    def confusion_matrices(self, Y_test, Ye_test, nrows, ncols, figsize, title_size = None, label_size = None, titles = None):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        axes = axes.flatten()

        t_size = 18
        if title_size is not None:
            t_size = title_size

        l_size = 13
        if label_size is not None:
            l_size = label_size

        for i in range(len(Ye_test)):
            if titles is not None:
                axes[i].set_title(titles[i], fontsize=t_size)

            axes[i].tick_params(axis="both", labelsize=l_size)
            axes[i].set_xlabel("Predicted labels", fontsize=l_size)
            axes[i].set_ylabel("True labels", fontsize=l_size)

            ConfusionMatrixDisplay.from_predictions(Y_test[i], Ye_test[i], ax=axes[i])

        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.35)

    def local_shap(self, explanations, ebm_data, nrows, ncols, figsize, title_size, label_size, titles, ebm_colors,
                   num_features=None):

        for i in range(len(explanations)):
            ax = plt.subplot(nrows, ncols, i + 1)
            ax.set_title(titles[i], fontsize=title_size)
            ax.tick_params(axis="both", labelsize=label_size)
            if num_features is not None:
                shap.plots.waterfall(explanations[i], show=False, max_display=num_features)
            else:
                shap.plots.waterfall(explanations[i], show=False)

        ax = plt.subplot(nrows, ncols, len(explanations) + 1)
        # Consider just the features importances without the interaction between them
        keep_indexes = [i for i in range(len(ebm_data["names"])) if "&" not in ebm_data["names"][i]]
        ebm_features_importances = np.array(ebm_data["scores"])[keep_indexes]
        ebm_feature_names = np.array(ebm_data["names"])[keep_indexes]

        # Sort by importance
        sorted_idxs = np.argsort(np.abs(ebm_features_importances))[::-1]
        ebm_importances_sorted = ebm_features_importances[sorted_idxs]
        ebm_names_sorted = ebm_feature_names[sorted_idxs]

        if num_features is not None:
            sum_other_importances = np.sum(ebm_importances_sorted[num_features:])

            ebm_importances_sorted = ebm_importances_sorted[:num_features]
            ebm_names_sorted = ebm_names_sorted[:num_features]

            ebm_importances_sorted = np.append(ebm_importances_sorted, sum_other_importances)
            ebm_names_sorted = np.append(ebm_names_sorted, "Sum of Rest")

        ebm_names_sorted = ebm_names_sorted[::-1]
        ebm_importances_sorted = ebm_importances_sorted[::-1]

        colors = [ebm_colors[0] if x < 0 else ebm_colors[1] for x in
                  ebm_importances_sorted]

        ax.barh(ebm_names_sorted, ebm_importances_sorted, color=colors)
        ax.set_title("Explainable Boosting Machine", fontsize=title_size, pad=28)
        ax.set_xlabel("Local Importance of each feature", fontsize=label_size)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", labelsize=label_size)
        ax.tick_params(axis="y", which="both", left=False)

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.25, hspace=0.35)

        plt.gcf().set_figwidth(figsize[0])
        plt.gcf().set_figheight(figsize[1])

    def global_shap(self, shap_values, X_test_pd, ebm_data, nrows, ncols, figsize, title_size, label_size, titles,
                    ebm_colors, num_features = None):
        plt.figure(figsize=figsize)

        for i in range(len(shap_values)):
            plt.subplot(nrows, ncols, i + 1)
            plt.title(titles[i], fontsize=title_size)
            if num_features is not None:
                shap.summary_plot(shap_values[i], X_test_pd, plot_type="bar", show=False, plot_size=None,
                                  class_inds="original", max_display=num_features)
            else:
                shap.summary_plot(shap_values[i], X_test_pd, plot_type="bar", show=False, plot_size=None,
                                  class_inds="original")


        ax = plt.subplot(nrows, ncols, len(shap_values) + 1)

        # Consider just the features importances without the interaction between them
        keep_indexes = [i for i in range(len(ebm_data["names"])) if
                        "&" not in ebm_data["names"][i]]
        ebm_features_importances = np.array(ebm_data["scores"])[keep_indexes]
        ebm_feature_names = np.array(ebm_data["names"])[keep_indexes]

        # Sort by importance
        sorted_idxs = np.argsort(ebm_features_importances)[::-1]
        ebm_importances_sorted = ebm_features_importances[sorted_idxs]
        ebm_names_sorted = ebm_feature_names[sorted_idxs]

        if num_features is not None:
            ebm_importances_sorted = ebm_importances_sorted[:num_features]
            ebm_names_sorted = ebm_names_sorted[:num_features]

        ebm_names_sorted = ebm_names_sorted[::-1]
        ebm_importances_sorted = ebm_importances_sorted[::-1]

        ax.barh(ebm_names_sorted, ebm_importances_sorted, color=ebm_colors[0])
        ax.set_title("Explainable Boosting Machine", fontsize=title_size)
        ax.set_xlabel("Global Importance of each feature", fontsize=label_size)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=label_size)

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0.35)

    def acc_classes(self, cms, nrows, ncols, figsize, titles=None, labels=None, title_size=None, label_size=None,
                    colors=None):
        plt.figure(figsize=figsize)

        t_size = 18
        if title_size is not None:
            t_size = title_size

        l_size = 13
        if label_size is not None:
            l_size = label_size

        for i in range(len(cms)):
            ax = plt.subplot(nrows, ncols, i + 1)

            if titles is not None:
                ax.set_title(titles[i], fontsize=t_size)

            ax.tick_params(axis="both", labelsize=l_size)
            ax.set_xlabel("Accuracy (%)", fontsize=l_size)
            ax.set_ylabel("Classes", fontsize=l_size)

            labels_plt = labels
            if labels is None:
                labels_plt = range(len(cms[0]))

            self.perf_analysis.acc_for_class(cms[i],
                                        colors=colors[i],
                                        title=titles[i],
                                        labels=labels_plt,
                                        title_size=title_size,
                                        label_size=label_size)

        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0.35)


