import keras as keras
from keras import layers
from keras import models
from keras.optimizers import Adam

# For 1D data embeddings
def create_fully_connected(input_shape, task, n_classes = None):
    model = models.Sequential()
    model.add(layers.Dense(10, activation = "relu", input_shape=input_shape))
    model.add(layers.Dense(10, activation = "relu"))
    model.add(layers.Flatten())
    if task == "binary":
        model.add(layers.Dense(1, activation = "sigmoid" ))
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            model.add(layers.Dense(n_classes, activation = "softmax"))
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        model.add(layers.Dense(1))
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-3), loss=loss, metrics=[metric])
    return model

#For 2D data embeddings (non_reduced) and 1D data (time series, fft, physicochemical properties, etc.)
def create_conv1d(input_shape, task, n_classes = None):
    model = models.Sequential()
    model.add(layers.Conv1D(16, 5, activation = "relu", input_shape=input_shape))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(32, 5, activation = "relu"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation = "relu" ))
    if task == "binary":
        model.add(layers.Dense(1, activation = "sigmoid" ))
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            model.add(layers.Dense(n_classes, activation = "softmax"))
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        model.add(layers.Dense(1))
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-3), loss=loss, metrics=[metric])
    return model

def create_lstm(input_shape, task, n_classes = None):
    model = models.Sequential()
    model.add(layers.LSTM(32, input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.LSTM(32))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation = "relu" ))
    if task == "binary":
        model.add(layers.Dense(1, activation = "sigmoid" ))
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            model.add(layers.Dense(n_classes, activation = "softmax"))
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        model.add(layers.Dense(1))
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])
    return model

def create_gru(input_shape, task, n_classes = None):
    model = models.Sequential()
    model.add(layers.GRU(32, input_shape=input_shape, return_sequences=True))
    model.add(layers.GRU(32, return_sequences=True))
    model.add(layers.GRU(32))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation = "relu" ))
    if task == "binary":
        model.add(layers.Dense(1, activation = "sigmoid" ))
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            model.add(layers.Dense(n_classes, activation = "softmax"))
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        model.add(layers.Dense(1))
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])
    return model

def create_bilstm(input_shape, task, n_classes = None):
    model = models.Sequential()
    model.add(layers.Bidirectional(layers.LSTM(32), input_shape=input_shape))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation = "relu" ))
    if task == "binary":
        model.add(layers.Dense(1, activation = "sigmoid" ))
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            model.add(layers.Dense(n_classes, activation = "softmax"))
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        model.add(layers.Dense(1))
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])
    return model

def create_bigru(input_shape, task, n_classes = None):
    model = models.Sequential()
    model.add(layers.Bidirectional(layers.GRU(32), input_shape=input_shape))
    model.add(layers.Dense(10, activation = "relu" ))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation = "sigmoid" ))
    if task == "binary":
        model.add(layers.Dense(1, activation = "sigmoid" ))
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            model.add(layers.Dense(n_classes, activation = "softmax"))
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        model.add(layers.Dense(1))
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])
    return model


def create_conv1d_lstm(input_shape, task, n_classes = None):
    model = models.Sequential()
    model.add(layers.Conv1D(16, 5, activation = "relu", input_shape=input_shape))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 5, activation = "relu"))
    model.add(layers.LSTM(32))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation = "relu" ))
    if task == "binary":
        model.add(layers.Dense(1, activation = "sigmoid" ))
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            model.add(layers.Dense(n_classes, activation = "softmax"))
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        model.add(layers.Dense(1))
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])
    return model

def create_conv1d_bilstm(input_shape, task, n_classes = None):
    model = models.Sequential()
    model.add(layers.Conv1D(16, 5, activation = "relu", input_shape=input_shape))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 5, activation = "relu"))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation = "relu" ))
    if task == "binary":
        model.add(layers.Dense(1, activation = "sigmoid" ))
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            model.add(layers.Dense(n_classes, activation = "softmax"))
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        model.add(layers.Dense(1))
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])
    return model

def create_conv1d_gru(input_shape, task, n_classes = None):
    model = models.Sequential()
    model.add(layers.Conv1D(16, 5, activation = "relu", input_shape=input_shape))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 5, activation = "relu"))
    model.add(layers.GRU(32))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation = "relu" ))
    if task == "binary":
        model.add(layers.Dense(1, activation = "sigmoid" ))
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            model.add(layers.Dense(n_classes, activation = "softmax"))
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        model.add(layers.Dense(1))
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])
    return model

def create_conv1d_bigru(input_shape, task, n_classes = None):
    model = models.Sequential()
    model.add(layers.Conv1D(16, 5, activation = "relu", input_shape=input_shape))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 5, activation = "relu"))
    model.add(layers.Bidirectional(layers.GRU(32, recurrent_dropout = 0.2), input_shape=input_shape))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation = "relu" ))
    if task == "binary":
        model.add(layers.Dense(1, activation = "sigmoid" ))
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            model.add(layers.Dense(n_classes, activation = "softmax"))
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        model.add(layers.Dense(1))
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])
    return model

def create_conv1d_lstm_parallel(input_shape, task, n_classes = None):
    inputs  = layers.Input(input_shape)

    tower_1 = layers.Conv1D(16, (5), activation='relu')(inputs)
    tower_1 = layers.MaxPooling1D(5)(tower_1)
    tower_1 = layers.Dropout(0.2)(tower_1)
    tower_1 = layers.GlobalMaxPooling1D()(tower_1)

    tower_2 = layers.LSTM(32, return_sequences=True)(inputs)
    tower_2 = layers.LSTM(32, recurrent_dropout = 0.2, return_sequences=True)(tower_2)
    tower_2 = layers.LSTM(32)(tower_2)

    x       = layers.Concatenate(axis=1)([tower_1, tower_2])
    x       = layers.Dropout(0.3)(x)
    x       = layers.Flatten()(x)
    x       = layers.Dense(10, activation='relu')(x)

    if task == "binary":
        preds = layers.Dense(1, activation = "sigmoid" )(x)
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            preds = layers.Dense(n_classes, activation = "softmax")(x)
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        preds = layers.Dense(1)(x)
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")

    model = models.Model(inputs, preds)
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])

    return model

def create_conv1d_bilstm_parallel(input_shape, task, n_classes = None):
    inputs  = layers.Input(input_shape)

    tower_1 = layers.Conv1D(16, (5), activation='relu')(inputs)
    tower_1 = layers.MaxPooling1D(5)(tower_1)
    tower_1 = layers.Dropout(0.2)(tower_1)
    tower_1 = layers.GlobalMaxPooling1D()(tower_1)

    tower_2 = layers.Bidirectional(layers.LSTM(32, recurrent_dropout = 0.2))(inputs)
    x       = layers.Concatenate(axis=1)([tower_1, tower_2])
    x       = layers.Flatten()(x)
    x       = layers.Dropout(0.3)(x)
    x       = layers.Dense(10, activation='relu')(x)
    if task == "binary":
        preds = layers.Dense(1, activation = "sigmoid" )(x)
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            preds = layers.Dense(n_classes, activation = "softmax")(x)
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        preds = layers.Dense(1)(x)
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")

    model = models.Model(inputs, preds)
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])

    return model

def create_conv1d_gru_parallel(input_shape, task, n_classes = None):
    inputs  = layers.Input(input_shape)

    tower_1 = layers.Conv1D(16, (5), activation='relu')(inputs)
    tower_1 = layers.MaxPooling1D(5)(tower_1)
    tower_1 = layers.Dropout(0.2)(tower_1)
    tower_1 = layers.GlobalMaxPooling1D()(tower_1)

    tower_2 = layers.GRU(32, return_sequences=True)(inputs)
    tower_2 = layers.GRU(32, recurrent_dropout = 0.2, return_sequences=True)(tower_2)
    tower_2 = layers.GRU(32)(tower_2)

    x       = layers.Concatenate(axis=1)([tower_1, tower_2])
    x       = layers.Flatten()(x)
    x       = layers.Dense(10, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)

    if task == "binary":
        preds = layers.Dense(1, activation = "sigmoid" )(x)
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            preds = layers.Dense(n_classes, activation = "softmax")(x)
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        preds = layers.Dense(1)(x)
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")

    model = models.Model(inputs, preds)
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])

    return model

def create_conv1d_bigru_parallel(input_shape, task, n_classes = None):
    
    inputs  = layers.Input(input_shape)

    tower_1 = layers.Conv1D(16, (5), activation='relu')(inputs)
    tower_1 = layers.MaxPooling1D(5)(tower_1)
    tower_1 = layers.Dropout(0.2)(tower_1)
    tower_1 = layers.GlobalMaxPooling1D()(tower_1)

    tower_2 = layers.Bidirectional(layers.GRU(32, recurrent_dropout = 0.2))(inputs)

    x       = layers.Concatenate(axis=1)([tower_1, tower_2])
    x       = layers.Flatten()(x)
    x       = layers.Dropout(0.3)(x)
    x       = layers.Dense(10, activation='relu')(x)
    
    if task == "binary":
        preds = layers.Dense(1, activation = "sigmoid" )(x)
        loss = "binary_crossentropy"
        metric = "acc"
    elif task == "categorical":
        if n_classes:
            preds = layers.Dense(n_classes, activation = "softmax")(x)
            loss = "categorical_crossentropy"
            metric = "acc"
        else:
            raise ValueError("Needs n_class.")
    elif task == "regression":
        preds = layers.Dense(1)(x)
        loss = "mean_squared_error"
        metric = "mean_squared_error"
    else:
        raise ValueError(f"Task {task} not allowed")

    model = models.Model(inputs, preds)
    model.summary()
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss=loss, metrics=[metric])

    return model
