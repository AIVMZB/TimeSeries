from datapreparation import get_test_series
from tensorflow import keras

from model import forecast


def main():
    series = get_test_series()
    model = keras.models.load_model("trained_model")
    forecast(model, series)


if __name__ == '__main__':
    main()
