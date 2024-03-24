import config as conf
from tensorflow import keras
import datapreparation as dp
import matplotlib.pyplot as plt
import numpy as np

window_size = conf.TS_WINDOW_SIZE
path = "trained_model"


def plot_forecast(series, forecast, zoom=True):
    if zoom:
        series = series[-len(forecast):]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(series)), series, label='Original Series')
    plt.plot(range(len(series), len(series) + len(forecast)), forecast, label='Forecasted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Time Series Forecasting')
    plt.legend()
    plt.show()


def get_model():
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=3,
                               strides=1,
                               activation="relu",
                               padding='causal',
                               input_shape=[window_size, 1]),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.compile(keras.optimizers.Adam(0.001), keras.losses.Huber(),
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

    history = model.fit(train_data, epochs=85)

    model.save(path)

    plot_history(history.history)


def forecast(model: keras.Model, series, n_steps: int = 100, plot=True, zoom=True):
    x = series[-window_size:]
    x = x.reshape(1, window_size, 1)

    predictions = []

    for _ in range(n_steps):
        next_step = model.predict(x)
        x = np.concatenate([x[:, 1:, :], next_step.reshape(1, 1, 1)], axis=1)
        predictions.append(next_step[0, 0])

    if plot:
        plot_forecast(series, predictions, zoom)

    return predictions


if __name__ == '__main__':
    train_model(
        get_model()
    )
