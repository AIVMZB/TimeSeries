import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from datapreparation import get_train_windowed_data
from model import get_model


def adjust_lr(model: keras.Model, data: np.ndarray):
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10 ** (epoch / 20)
    )

    history = model.fit(data, epochs=5, callbacks=[lr_schedule])

    lrs = 1e-8 * (10 ** (np.arange(5) / 20))

    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Set the grid
    plt.grid(True)

    # Plot the loss in log scale
    plt.semilogx(lrs, history.history["loss"])

    # Increase the tickmarks size
    plt.tick_params('both', length=10, width=1, which='both')


if __name__ == '__main__':
    model = get_model()
    data = get_train_windowed_data()

    adjust_lr(model, data)
