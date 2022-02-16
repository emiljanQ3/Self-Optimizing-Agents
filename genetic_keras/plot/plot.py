from genetic_keras.data import EpochData
from genetic_keras.utils import set_params
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np





def x_sum_plot(epoch_history, fig, ax, train_x, train_y, val_x, val_y, model):
    epoch_hist_plot(epoch_history, ax[0], "Loss over epochs")

    x_sum_data_fit_plot(model, train_x, train_y, ax[1], "Best fit on training data in population")

    x_sum_data_fit_plot(model, val_x, val_y, ax[2], "Best training chromosome fit on validation data")

    plt.pause(0.000000000001)


def x_sum_data_fit_plot(model, x, y_target, ax, title):
    y_prediction = model(x).numpy()
    x_sum = np.sum(x, 1)

    ax.clear()

    ax.scatter(x_sum, y_target, label="Target values")
    ax.scatter(x_sum, y_prediction, label="Predicted values")

    ax.set_xlabel("Summed input")
    ax.set_ylabel("Function output")
    ax.set_title(title)
    ax.legend()


def epoch_hist_plot(epoch_history, ax, title):
    epochs = [d.index for d in epoch_history]
    train_loss_best = [-d.best_loss_training for d in epoch_history]
    train_loss_median = [-d.median_loss_training for d in epoch_history]

    ax.clear()

    final_best = "{:.2e}".format(train_loss_best[-1])
    final_median = "{:.2e}".format(train_loss_median[-1])
    ax.plot(train_loss_median, "-b", label="Median training score." + final_median)
    ax.plot(train_loss_best, "--b", label="Best training score " + final_best)

    ax.set_xlabel("Generation index")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
