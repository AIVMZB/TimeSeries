import config as conf
from tensorflow import keras
import datapreparation as dp
import matplotlib.pyplot as plt

window_size = conf.TS_WINDOW_SIZE
path = "trained_model"


def get_model():
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=3,
                            strides=1, activation="relu",
                            padding='causal', input_shape=[window_size, 1]),
        keras.layers.GRU(64, activation="relu", return_sequences=True),
        keras.layers.GRU(128, activation="relu"),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.compile(keras.optimizers.Adam(0.01), keras.losses.Huber(),
                  metrics=["mse", "mae"])

    return model


def plot_history(history: dict):
    plt.plot(history["loss"], label="Train loss")
    plt.plot(history["mae"], label="Train MAE")
    plt.plot(history["mse"], label="Train MSE")
    plt.legend()
    plt.show()


def train_model(model: keras.Model):
    train_data = dp.get_train_windowed_data()

    history = model.fit(train_data, epochs=8)

    model.save(path)

    plot_history(history.history)


if __name__ == '__main__':
    train_model(
        get_model()
    )
