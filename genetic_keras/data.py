import numpy as np


class EpochData:
    def __init__(self,
                 time,
                 best_chromosome_training,
                 best_loss_training,
                 median_loss_training,
                 best_chromosome_validation,
                 best_loss_validation,
                 median_loss_validation,
                 index,
                 best_accuracy_training,
                 median_accuracy_training,
                 best_accuracy_validation,
                 median_accuracy_validation,
                 ):
        self.time = time
        self.best_chromosome_training = best_chromosome_training
        self.best_loss_training = best_loss_training
        self.median_loss_training = median_loss_training
        self.best_chromosome_validation = best_chromosome_validation
        self.best_loss_validation = best_loss_validation
        self.median_loss_validation = median_loss_validation
        self.index = index
        self.best_accuracy_training = best_accuracy_training
        self.median_accuracy_training = median_accuracy_training
        self.best_accuracy_validation = best_accuracy_validation
        self.median_accuracy_validation = median_accuracy_validation


def generate_epoch_data(population: np.ndarray, train_losses: np.ndarray, validation_losses: np.ndarray, epoch_time,
                        index, train_accuracy, validation_accuracy):
    if train_accuracy is not None:
        best_train_acc_idx = np.argmax(train_accuracy)
        best_val_acc_idx = np.argmax(validation_accuracy)

        best_accuracy_training = train_accuracy[best_train_acc_idx]
        median_accuracy_training = np.median(train_accuracy)
        best_accuracy_validation = validation_accuracy[best_val_acc_idx]
        median_accuracy_validation = np.median(validation_accuracy)

    else:
        best_accuracy_training = None
        median_accuracy_training = None
        best_accuracy_validation = None
        median_accuracy_validation = None

    best_train_loss_idx = np.argmin(train_losses)
    best_val_loss_idx = np.argmin(validation_losses)
    return EpochData(time=epoch_time,
                     best_chromosome_training=population[best_train_loss_idx],
                     best_loss_training=train_losses[best_train_loss_idx],
                     median_loss_training=np.median(train_losses),
                     best_chromosome_validation=population[best_val_loss_idx],
                     best_loss_validation=validation_losses[best_val_loss_idx],
                     median_loss_validation=np.median(validation_losses),
                     index=index,
                     best_accuracy_training=best_accuracy_training,
                     median_accuracy_training=median_accuracy_training,
                     best_accuracy_validation=best_accuracy_validation,
                     median_accuracy_validation=median_accuracy_validation,
                     )
