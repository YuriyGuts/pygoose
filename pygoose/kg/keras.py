import matplotlib.pyplot as plt


def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples.

    Args:
        y: A list of true labels (the labels must be hashable).
        smooth_factor: A factor that smooths extremely uneven weights.

    Returns:
        A dictionary with the weight for each class.
    """

    from collections import Counter
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}


def plot_loss_history(history, figsize=(15, 8)):
    """
    Plots the learning history for a Keras model,
    assuming the validation data was provided to the 'fit' function.

    Args:
        history: The return value from the 'fit' function.
        figsize: The size of the plot.
    """

    plt.figure(figsize=figsize)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])

    plt.xlabel("# Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss over time")

    plt.show()


def plot_accuracy_history(history, figsize=(15, 8)):
    """
    Plots the learning history for a Keras model,
    assuming the validation data was provided to the 'fit' function.

    Args:
        history: The return value from the 'fit' function.
        figsize: The size of the plot.
    """

    plt.figure(figsize=figsize)

    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])

    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Validation"])
    plt.title("Accuracy over time")

    plt.show()
