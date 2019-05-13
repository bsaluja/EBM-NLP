import time
import sys
import numpy as np

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

def plot_bar_graph(x_data_points, y_data_points, x_axis_label, y_axis_label, graph_title, filepath):
    # objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
    y_pos = np.arange(len(x_data_points))
    # performance = [10,8,6,4,2,1]

    plt.bar(y_pos, y_data_points, align='center', alpha=0.5)
    plt.xticks(y_pos, x_data_points)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(graph_title)
    # plt.show()
    print("saving bar plot at location: ", filepath)
    plt.savefig(filepath)

def plot_bar_graph_for_two_data_series(x_data_points, y1_data_points, y2_data_points, y1_legend, y2_legend, x_axis_label, y_axis_label, graph_title, filepath):
    x_data_points = x_data_points[:30]
    y1_data_points = y1_data_points[:30]
    y2_data_points = y2_data_points[:30]
    # data to plot
    n_groups = len(y1_data_points)  #len(y1_data_points) should be equal to len(y2_data_points)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.5
    opacity = 0.8

    rects1 = plt.bar(index, y1_data_points, bar_width,
    alpha=opacity,
    color='b',
    label=y1_legend)

    rects2 = plt.bar(index + bar_width, y2_data_points, bar_width,
    alpha=opacity,
    color='g',
    label=y2_legend)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(graph_title)
    plt.xticks(index + bar_width, x_data_points)
    plt.legend()
    #set parameters for tick labels
    plt.tick_params(axis='x', which='major', labelsize=3)

    plt.tight_layout()
    # plt.show()
    print("saving bar plot at location: ", filepath)
    plt.savefig(filepath)


def plot_confusion_graph(self, confusion, all_tags, filepath):
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_tags, rotation=90)
    ax.set_yticklabels([''] + all_tags)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    # f = self.config.fig_confusionplot+'_' + str(epoch) +'.png'

    print("saving confusion plot at location: ", filepath)
    plt.savefig(filepath)
    # plt.savefig(self.config.fig_confusionplot, dpi = 300)


