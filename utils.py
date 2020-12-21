from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import matplotlib.pyplot as plt


def load_data_with_label(path_arr, data_arr, label_arr, img_size, _label):
    for path in path_arr:
        try:
            img = load_img(path, target_size = img_size)
            img = img_to_array(img)
            img = preprocess_input(img)
            data_arr.append(img)
            label_arr.append(_label)
        except:
            print(path)

def training_log_plot(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8,8))
    plt.subplot(2, 1, 1)
    plt.plot(accuracy, label = "TRNG_ACC")
    plt.plot(val_accuracy, label = "VAL_ACC")

    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loss, label = "TRNG_LOSS")
    plt.plot(val_loss, label = "VAL_LOSS")

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.title("Loss")
    plt.legend()

    plt.savefig("./training_log/fig.png")
    plt.show()

def export_model_summary(model, filename):
    with open("./model_summary/" + filename + ".txt", "w") as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))