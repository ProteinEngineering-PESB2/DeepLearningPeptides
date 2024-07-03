
import numpy as np
import keras as keras
from architectures import *
from utils import process_data, create_callbacks
import sys

from keras.utils import set_random_seed

X_path = sys.argv[1]
y_path = sys.argv[2]
architecture = sys.argv[3]
model_name = sys.argv[4]
seed = int(sys.argv[5])
if seed is None:
    seed = 42

set_random_seed(42)

callbacks_list = create_callbacks(model_name, seed, checkpoint_path = None)

X = np.load(X_path)
y = np.load(y_path)
X_train, X_test, y_train, y_test = process_data(X, y, seed, 0.3, use_rus=True)
input_shape = X_train.shape[1:]
model = create_conv1d(input_shape, "binary")

history = model.fit(X_train, y_train, validation_data = (X_test, y_test),
                    batch_size=20, epochs=100, callbacks=callbacks_list)
print(history)