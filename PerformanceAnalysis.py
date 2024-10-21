import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
import pandas as pd


class PerformanceAnalysis:
    def __init__(self, def_title_fs = None, def_axis_fs = None):
        """
        :param def_title_fs: default font size of the title
        :param def_axis_fs: default font size of the axis
        """
        if def_title_fs is not None:
            self.def_title_fs = def_title_fs
        else:
            self.def_title_fs = 18

        if def_axis_fs is not None:
            self.def_axis_fs = def_axis_fs
        else:
            self.def_axis_fs = 10


    def plot_PR(self, ytest_b, cm, y2d, title_font_size = None, axis_font_size = None):
        """
        :param ytest_b: Array with labels of test data
        :param cm: Confusion Matrix
        :param y2d: The classifier threshold
        :param title_fs: Font Size of the title
        :param axis_fs: Font Size of both axis
        """
        title_fs = self.def_title_fs
        if title_font_size is not None:
            title_fs = title_font_size

        axis_fs = self.def_axis_fs
        if axis_font_size is not None:
            axis_fs = axis_font_size


        pr, rc, t = precision_recall_curve(ytest_b, y2d)

        plt.plot(pr, rc, "r")
        plt.title("Precision-Recall (PR)", fontsize=title_fs)
        plt.axis('scaled')
        plt.axis((-.01, 1.01, -.01, 1.01))
        plt.tick_params(axis='both', labelsize=axis_fs)
        plt.grid(True)

        plt.xlabel("Recall")
        plt.ylabel("Precision")


        p10 = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        r10 = cm[1, 1] / (cm[1, 1] + cm[1, 0])

        plt.plot(p10, r10, 'ok')


    def compare_PR(self, ytests, cms, y2ds, labels=None, title_font_size= None, axis_font_size= None):
        """
        :param ytests: Array with labels of each classifier
        :param cms: Array with confusion matrixes of each classifier
        :param y2ds: Array with thresholds of each classifier
        :param labels: Label of each classifier
        """
        title_fs = self.def_title_fs
        if title_font_size is not None:
            title_fs = title_font_size

        axis_fs = self.def_axis_fs
        if axis_font_size is not None:
            axis_fs = axis_font_size


        plt.title("Precision-Recall (PR)", fontsize=title_fs)
        plt.axis('scaled')
        plt.axis((-.01, 1.01, -.01, 1.01))
        plt.tick_params(axis='both', labelsize=axis_fs)
        plt.grid(True)

        plt.xlabel("Recall")
        plt.ylabel("Precision")

        for idx in range(len(cms)):

            pr, rc, t = precision_recall_curve(ytests[idx], y2ds[idx])

            if labels is not None:
                plt.plot(pr, rc, label=labels[idx])
            else:
                plt.plot(pr, rc)

            cm = cms[idx]

            p10 = cm[1, 1] / (cm[1, 1] + cm[0, 1])
            r10 = cm[1, 1] / (cm[1, 1] + cm[1, 0])

            plt.plot(p10, r10, 'ok')

        if labels is not None:
            plt.legend()


    def plot_ROC(self, ytest_b, cm, y2d, title_font_size = None, axis_font_size = None):
        """
        :param ytest_b: Array with labels of test data
        :param cm: Confusion Matrix
        :param y2d: The classifier threshold
        :param title_fs: Font Size of the title
        :param axis_fs: Font Size of both axis
        """
        fp1, tp1, l1 = roc_curve(ytest_b, y2d)

        title_fs = self.def_title_fs
        if title_font_size is not None:
            title_fs = title_font_size

        axis_fs = self.def_axis_fs
        if axis_font_size is not None:
            axis_fs = axis_font_size

        plt.plot(fp1, tp1, 'r')
        plt.title("Receiver Operating Characteristic (ROC)", fontsize=title_fs)
        plt.axis('scaled')
        plt.axis((-.01, 1.01, -.01, 1.01))
        plt.tick_params(axis='both', labelsize=axis_fs)
        plt.grid(True)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        fp10 = cm[0, 1] / (cm[0, 1] + cm[0, 0])
        tp10 = cm[1, 1] / (cm[1, 1] + cm[1, 0])

        plt.plot(fp10, tp10, 'ok')


    def compare_ROC(self, ytests, cms, y2ds, labels=None, title_font_size = None, axis_font_size = None):

        title_fs = self.def_title_fs
        if title_font_size is not None:
            title_fs = title_font_size

        axis_fs = self.def_axis_fs
        if axis_font_size is not None:
            axis_fs = axis_font_size

        plt.title("Receiver Operating Characteristic (ROC)", fontsize=title_fs)
        plt.axis('scaled')
        plt.axis((-.01, 1.01, -.01, 1.01))
        plt.tick_params(axis='both', labelsize=axis_fs)
        plt.grid(True)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        for idx in range(len(cms)):
            fp1, tp1, l1 = roc_curve(ytests[idx], y2ds[idx])

            if labels is not None:
                plt.plot(fp1, tp1, label=labels[idx])
            else:
                plt.plot(fp1, tp1)

            cm = cms[idx]

            fp10 = cm[0, 1] / (cm[0, 1] + cm[0, 0])
            tp10 = cm[1, 1] / (cm[1, 1] + cm[1, 0])

            plt.plot(fp10, tp10, 'ok')

        if labels is not None:
            plt.legend()


    def compute_metrics(self, cm):
        tp = cm[1, 1]
        fp = cm[0, 1]
        tn = cm[0, 0]
        fn = cm[1, 0]

        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = (tp) / (tp + fp)
        recall = (tp) / (tp + fn)
        f_score = (2 * precision * recall) / (precision + recall)
        fp_rate = (fp) / (fp + tn)
        fn_rate = (fn) / (tp + fn)

        metrics = {"accuracy": accuracy,
                   "false-positive rate": fp_rate,
                   "false-negative rate": fn_rate,
                   "precision": precision,
                   "recall": recall,
                   "f-score": f_score,
                   }
        pandas_dict = pd.DataFrame(data=metrics, index=["Values"]).T
        pandas_dict.columns.names = ["Metrics"]

        return pandas_dict

    def acc_for_class(self, cm, figure=False, labels=None, colors=None, title=None, title_size=None, label_size=None):
        acc_percent = []
        for row_idx in range(len(cm)):
            acc_percent.append(cm[row_idx][row_idx] / np.sum(cm[row_idx]))

        acc_percent_round = [round(x * 100, 1) for x in acc_percent]
        bar_text_labels = [f"{x}" for x in acc_percent_round]

        if figure == True:
            title_plt = "Accuracy of Each Class"
            if title is not None:
                title_plt = title

            title_s = 18
            if title_size is not None:
                title_s = title_size

            label_s = 13
            if label_size is not None:
                label_s = label_size

            if labels is not None:
                container = plt.bar(labels, acc_percent_round, color=colors)
            else:
                container = plt.bar(range(len(cm)), acc_percent_round, color=colors)

            plt.bar_label(container, bar_text_labels, label_type="edge", fontsize=label_s)

            plt.title(title_plt, fontsize=title_s, pad=20)

            plt.xlabel("Class", fontsize=label_s)
            plt.ylabel("Accuracy (%)", fontsize=label_s)
            plt.xticks(fontsize=label_s)
            plt.yticks(fontsize=label_s)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)

            plt.tight_layout()

        acc = np.sum(np.diag(cm)) / np.sum(cm.flatten())
        pd_accs = pd.DataFrame(acc_percent)
        pd_accs.columns.names = ["Classes"]
        pd_accs.loc[len(cm)] = np.mean(pd_accs.iloc[:, 0])
        pd_accs = pd_accs.rename(index={len(cm): "Average"})
        pd_accs.loc[len(cm)] = acc
        pd_accs = pd_accs.rename(index={len(cm): "Accuracy"})


        return pd_accs
