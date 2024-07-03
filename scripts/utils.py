import keras as keras
import os
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

EARLY_STOP_PATIENCE = 10

def create_callbacks(history_path, model_path):
    callbacks = [
    keras.callbacks.EarlyStopping(
        monitor = "recall",
        mode = "max",
        patience = EARLY_STOP_PATIENCE,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_acc",
        factor=0.1,
        patience=5),
    keras.callbacks.CSVLogger(
        history_path,
        separator=",",
        append=False),
    keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_acc",
        save_best_only=True
        )
    ]
    return callbacks

def process_data(data, target, random_state, test_size, use_rus = True, sample_size = None):
    #Random under sampling
    index = np.arange(0, data.shape[0])
    if use_rus:
        rus = RandomUnderSampler(random_state=random_state, replacement=True)
        index_data, target = rus.fit_resample(index.reshape(-1,1), target)
        data = data[np.ndarray.flatten(index_data)]
    if sample_size is not None:
        idx = np.random.randint(data.shape[0], size= sample_size)
        data = data[idx]
        target = target[idx]
    #Train test split
    index = np.arange(0, data.shape[0])
    index_X_train, index_X_test, y_train, y_test = train_test_split(
        index, target, test_size=test_size, random_state=random_state)
    X_train = data[index_X_train]
    X_test = data[index_X_test]
    return X_train, X_test, y_train, y_test

def plot_results(history,):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    
    sns.set_theme()
    plt.figure(figsize=(6.4,9.6))
    plt.subplot(2,1,1)
    plt.plot(epochs, acc, 'bo', label="Training acc")
    plt.plot(epochs, val_acc, 'b', label="Validation acc")
    plt.axvline(x = epochs[-1] - EARLY_STOP_PATIENCE, color = 'r', linestyle="dashed")
    plt.xlabel("Epochs")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(epochs, loss, 'bo', label="Training loss")
    plt.plot(epochs, val_loss, 'b', label="Validation loss")
    plt.axvline(x = epochs[-1] - EARLY_STOP_PATIENCE, color = 'r', linestyle="dashed")
    plt.xlabel("Epochs")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
