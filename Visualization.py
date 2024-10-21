import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2
from PerformanceAnalysis import PerformanceAnalysis
from GradCAM import GradCAM
class Visualization:

    def __init__(self):
        self.perf_analysis = PerformanceAnalysis()
        self.gradcam = GradCAM()


    def confusion_matrices_pred(self, Y_test, Ye_test, nrows, ncols, figsize, title_size = None, label_size = None, titles = None, save_name=None):

        t_size = 18
        if title_size is not None:
            t_size = title_size

        l_size = 13
        if label_size is not None:
            l_size = label_size

        if save_name is not None:
            for i in range(len(Ye_test)):

                fig = plt.figure(figsize=figsize)

                ConfusionMatrixDisplay.from_predictions(Y_test[i], Ye_test[i], ax=plt.gca())

                plt.gca().tick_params(axis="both", labelsize=l_size)
                plt.gca().set_xlabel("Predicted labels", fontsize=l_size)
                plt.gca().set_ylabel("True labels", fontsize=l_size)

                if titles is not None:
                    plt.gca().set_title(titles[i], fontsize=t_size)

                plt.tight_layout()

                fig.savefig(f"{save_name}_{i}.png", bbox_inches="tight")

        else:
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

            axes = axes.flatten()

            for i in range(len(Ye_test)):
                if titles is not None:
                    axes[i].set_title(titles[i], fontsize=t_size)

                axes[i].tick_params(axis="both", labelsize=l_size)
                axes[i].set_xlabel("Predicted labels", fontsize=l_size)
                axes[i].set_ylabel("True labels", fontsize=l_size)

                ConfusionMatrixDisplay.from_predictions(Y_test[i], Ye_test[i], ax=axes[i])

            plt.tight_layout()
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.35)

    def confusion_matrices(self, cms, nrows, ncols, figsize, title_size=18, label_size=13, titles=None, save_name=None):

        if save_name is not None:
            for i in range(len(cms)):

                fig = plt.figure(figsize=figsize)

                cm_disp = ConfusionMatrixDisplay(confusion_matrix=cms[i])

                plt.gca().tick_params(axis="both", labelsize=label_size)
                plt.gca().set_xlabel("Predicted labels", fontsize=label_size)
                plt.gca().set_ylabel("True labels", fontsize=label_size)

                if titles is not None:
                    plt.gca().set_title(titles[i], fontsize=title_size)

                plt.tight_layout()

                cm_disp.plot(ax=plt.gca())

                fig.savefig(f"{save_name}_{i}.png", bbox_inches="tight")

        else:
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

            axes = axes.flatten()

            for i in range(len(cms)):
                if titles is not None:
                    axes[i].set_title(titles[i], fontsize=title_size)

                axes[i].tick_params(axis="both", labelsize=label_size)
                axes[i].set_xlabel("Predicted labels", fontsize=label_size)
                axes[i].set_ylabel("True labels", fontsize=label_size)

                disp = ConfusionMatrixDisplay(cms[i])
                disp.plot(ax=axes[i])

            plt.tight_layout()
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.35)


    def local_shap(self, explanations, ebm_data, nrows=2, ncols=2, figsize=(15,10), title_size=18, label_size=13, titles=None, ebm_colors=None,
                   num_features=None, save_name=None):

        for i in range(len(explanations)):
            if save_name is None:
                ax = plt.subplot(nrows, ncols, i + 1)
                ax.set_title(titles[i], fontsize=title_size)
                ax.tick_params(axis="both", labelsize=label_size)
            else:
                plt.figure(figsize=figsize)
                plt.title(titles[i], fontsize=title_size)
                plt.tick_params(axis="both", labelsize=label_size)

            if num_features is not None:
                shap.plots.waterfall(explanations[i], show=False, max_display=num_features)
            else:
                shap.plots.waterfall(explanations[i], show=False)

            if save_name is not None:
                plt.savefig(f"{save_name}_{i}.png", bbox_inches="tight")

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.25, hspace=0.35)
        plt.gcf().set_figwidth(figsize[0])
        plt.gcf().set_figheight(figsize[1])


        if ebm_data is not None:

            # Consider just the features importances without the interaction between them
            keep_indexes = [i for i in range(len(ebm_data["names"])) if "&" not in ebm_data["names"][i]]
            ebm_features_importances = np.array(ebm_data["scores"])[keep_indexes]
            ebm_feature_names = np.array(ebm_data["names"])[keep_indexes]

            # Sort by importance
            sorted_idxs = np.argsort(np.abs(ebm_features_importances))[::-1]
            ebm_importances_sorted = ebm_features_importances[sorted_idxs]
            ebm_names_sorted = ebm_feature_names[sorted_idxs]

            if num_features is not None and len(keep_indexes) > num_features:
                sum_other_importances = np.sum(ebm_importances_sorted[num_features-1:])

                ebm_importances_sorted = ebm_importances_sorted[:num_features-1]
                ebm_names_sorted = ebm_names_sorted[:num_features-1]

                ebm_importances_sorted = np.append(ebm_importances_sorted, sum_other_importances)
                ebm_names_sorted = np.append(ebm_names_sorted, "Sum of Rest")

            ebm_names_sorted = ebm_names_sorted[::-1]
            ebm_importances_sorted = ebm_importances_sorted[::-1]

            if ebm_colors is None:
                ebm_colors=[(0, 139, 251),(255, 0, 79)]

            colors = [ebm_colors[0] if x < 0 else ebm_colors[1] for x in
                      ebm_importances_sorted]

            if save_name is not None:
                plt.figure(figsize=figsize)
                plt.title("Explainable Boosting Machine", fontsize=title_size)
                plt.tick_params(axis="both", labelsize=label_size)

                plt.barh(ebm_names_sorted, ebm_importances_sorted, color=colors)

            else:
                ax = plt.subplot(nrows, ncols, len(explanations) + 1)

                ax.barh(ebm_names_sorted, ebm_importances_sorted, color=colors)
                ax.set_title("Explainable Boosting Machine", fontsize=title_size, pad=28)

            plt.gca().set_xlabel("Local Importance of each feature", fontsize=label_size)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().tick_params(axis="both", labelsize=label_size)
            plt.gca().tick_params(axis="y", which="both", left=False)

        if save_name is not None:
            plt.gcf().savefig(f"{save_name}_{len(explanations)}.png", bbox_inches="tight")


    def global_shap(self, shap_values, X_test_pd, ebm_data, nrows, ncols, figsize, title_size, label_size, titles,
                    ebm_colors, num_features = None, save_name=None):

        if save_name is None:
            plt.figure(figsize=figsize)

        for i in range(len(shap_values)):
            if save_name is None:
                ax = plt.subplot(nrows, ncols, i + 1)
                ax.set_title(titles[i], fontsize=title_size)
                ax.tick_params(axis="both", labelsize=label_size)
            else:
                plt.figure(figsize=figsize)
                plt.title(titles[i], fontsize=title_size)
                plt.tick_params(axis="both", labelsize=label_size)

            if num_features is not None:
                shap.summary_plot(shap_values[i], X_test_pd, plot_type="bar", show=False, plot_size=None,
                                  class_inds="original", max_display=num_features)
            else:
                shap.summary_plot(shap_values[i], X_test_pd, plot_type="bar", show=False, plot_size=None,
                                  class_inds="original")

            if save_name is not None:
                plt.savefig(f"{save_name}_{i}.png", bbox_inches="tight")


            # Consider just the features importances without the interaction between them
        keep_indexes = [i for i in range(len(ebm_data["names"])) if
                        "&" not in ebm_data["names"][i]]
        ebm_features_importances = np.array(ebm_data["scores"])[keep_indexes]
        ebm_feature_names = np.array(ebm_data["names"])[keep_indexes]

        # Sort by importance
        sorted_idxs = np.argsort(ebm_features_importances)[::-1]
        ebm_importances_sorted = ebm_features_importances[sorted_idxs]
        ebm_names_sorted = ebm_feature_names[sorted_idxs]

        if num_features is not None and len(keep_indexes) > num_features:
            ebm_importances_sorted = ebm_importances_sorted[:num_features]
            ebm_names_sorted = ebm_names_sorted[:num_features]

        ebm_names_sorted = ebm_names_sorted[::-1]
        ebm_importances_sorted = ebm_importances_sorted[::-1]

        if ebm_colors is None:
            ebm_colors = [(0, 139, 251), (255, 0, 79)]

        if save_name is not None:
            plt.figure(figsize=figsize)
            plt.title("Explainable Boosting Machine", fontsize=title_size)
            plt.tick_params(axis="both", labelsize=label_size)

            plt.barh(ebm_names_sorted, ebm_importances_sorted, color=ebm_colors[0])

        else:
            ax = plt.subplot(nrows, ncols, len(shap_values) + 1)

            ax.barh(ebm_names_sorted, ebm_importances_sorted, color=ebm_colors[0])
            ax.set_title("Explainable Boosting Machine", fontsize=title_size, pad=28)
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.25, hspace=0.35)

        plt.gca().set_xlabel("Global Importance of each feature", fontsize=label_size)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().tick_params(axis="y", labelsize=label_size)

        if save_name is not None:
            plt.gcf().savefig(f"{save_name}_{len(shap_values)}.png", bbox_inches="tight")


    def acc_classes(self, cms, nrows, ncols, figsize, titles=None, labels=None, title_size=18, label_size=13,
                    colors=None):

        plt.figure(figsize=figsize)

        for i in range(len(cms)):
            ax = plt.subplot(nrows, ncols, i + 1)

            if titles is not None:
                ax.set_title(titles[i], fontsize=title_size)

            ax.tick_params(axis="both", labelsize=label_size)
            ax.set_xlabel("Accuracy (%)", fontsize=label_size)
            ax.set_ylabel("Classes", fontsize=label_size)

            labels_plt = labels
            if labels is None:
                labels_plt = range(len(cms[0]))

            self.perf_analysis.acc_for_class(cms[i],
                                             figure=True,
                                            colors=colors[i],
                                            title=titles[i],
                                            labels=labels_plt,
                                            title_size=title_size,
                                            label_size=label_size)

        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0.35)

    def brain_tumor(self, images_to_explain, true_masks, model, last_conv_layer_name, save_name, lime_samples=None,
                    shap_evals=None, figsize=None, title_size=None):

        t_size = 12
        if title_size is not None:
            t_size = title_size

        f_size = (15, 10)
        if figsize is not None:
            f_size = figsize

        l_samples = 1000
        if lime_samples is not None:
            l_samples = lime_samples

        s_evals = 250
        if shap_evals is not None:
            s_evals = shap_evals

        for i in range(len(images_to_explain)):
            plt.figure(figsize=f_size)
            n_cols = 4
            if true_masks is None:
                n_cols = 5
            plt.subplot(1, n_cols, 1)
            plt.imshow(images_to_explain[i])
            plt.title("Original", fontsize=t_size)
            plt.axis("off")

            grad_cam = self.gradcam.get_gradcam_img(images_to_explain[i], model, last_conv_layer_name, alpha=0.4)

            explainer = lime_image.LimeImageExplainer(random_state=42)
            explanation = explainer.explain_instance(images_to_explain[i],
                                                     model.predict,
                                                     top_labels=3, hide_color=0, num_samples=l_samples,
                                                     random_seed=None)

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=True)

            lime_mask = mark_boundaries(temp, mask)
            lime_mask = cv2.normalize(lime_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            plt.subplot(1, n_cols, 2)
            plt.imshow(lime_mask)
            plt.title("LIME", fontsize=t_size)
            plt.axis("off")

            plt.subplot(1, n_cols, 3)
            plt.imshow(grad_cam)
            plt.title("Grad-CAM", fontsize=t_size)
            plt.axis("off")

            if true_masks is not None:
                plt.subplot(1, n_cols, 4)
                plt.imshow(true_masks[i])
                plt.title("True Mask", fontsize=t_size)
                plt.axis("off")

            plt.savefig(f"{save_name}_exp_{i}.png", bbox_inches="tight")

            # define a masker that is used to mask out partitions of the input image, this one uses a blurred background
            masker = shap.maskers.Image("inpaint_telea", images_to_explain[i].shape)

            explainer = shap.Explainer(model, masker, output_names=["SHAP", "SHAP", "SHAP"])

            shap_values = explainer(np.array([images_to_explain[i]]), max_evals=s_evals, batch_size=30,
                                    outputs=shap.Explanation.argsort.flip[:1])

            # Shap shows two images, so figsize is adjusted
            shap.image_plot(shap_values, show=False, width=int(f_size[0] / 3 * 2))
            plt.savefig(f"{save_name}_shap_{i}.png", bbox_inches="tight")



