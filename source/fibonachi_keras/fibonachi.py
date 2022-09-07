import numpy as np

from fibonachi_keras.fibonachi_model import mind_model


def fibonachi_sequence():
    sequence = [1, 1, 2, 3,
             5, 8, 13, 21,
             34, 55, 89, 144]

    array16 = []
    array16_in = []
    for i in range(len(sequence) - 3):
        array16.append(sequence[i:i + 4])
        array16_in.append(sequence[i:i + 3])

    model = mind_model()

    array16 = np.array(array16)
    array16_in = np.array(array16_in)

    print(array16_in[:-2])
    print(array16[:-2, -1])

    # SCALING AND NORMALIZATION!

    model.fit(array16_in[:-2], array16[:-2, -1], epochs=3500)

    predictions = model.predict(array16_in)

    delta = []

    for i in range(len(predictions)):
        if len(predictions) - 2 == i: print("-----Predict-----")
        delta.append(abs(array16[i][-1] - predictions[i]))
        print(array16[i][-1], predictions[i])

    print("Абсолютна похибка для усіх значень -", np.mean(delta))
