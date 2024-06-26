import os, argparse
import pandas as pd
import numpy as np
import keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from keras.utils import set_random_seed
from architectures import *
from utils import process_data, create_callbacks, plot_results
import sys

def get_metrics(model, X, labels, prefix: str) -> dict:
    y_pred = model.predict(X)
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]

    precision = precision_score(labels, y_pred, zero_division=0.0, average='binary')
    recall = recall_score(labels, y_pred, zero_division=0.0, average='binary')
    f1 = f1_score(labels, y_pred, zero_division=0.0, average='binary')
    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
    mcc = matthews_corrcoef(labels, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)

    cm_str = []
    for row in [[tn, fp], 
                [fn, tp]]:
        cm_str.append("[")
        for col in row:
            cm_str.extend([str(col), ","])
        cm_str = cm_str[:-1]
        cm_str.extend(["]", ","])
    
    cm_str = "".join(cm_str[:-1])
    
    metrics = {
            f'{prefix}precision': precision,
            f'{prefix}sensitivity': sensitivity,
            f'{prefix}recall': recall,
            f'{prefix}specificity': specificity,
            f'{prefix}f1-score': f1,
            f'{prefix}cm': cm_str,
            f'{prefix}mcc': mcc
    }

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="Path of the data", required=True)
    parser.add_argument("-d","--dataset", help="Name of the dataset", required=False)
    parser.add_argument("-e", "--encoding", help="Name of the encoding", required=True)
    parser.add_argument("-s","--seed", help ="Seed for training", required=False, default=42)
    parser.add_argument("-y", "--history_path", help="History path", required=True)
    parser.add_argument("-p", "--metrics_path", help="Metrics path", required=True)
    parser.add_argument("-m", "--model_path", help="Model path", required=True)
    parser.add_argument("--Dim2D", action="store_true")
    parser.add_argument("--nosplitted", help="If the data is not splitted, it splitts by the given seed", action="store_true")
    args = parser.parse_args()

    callbacks_list = create_callbacks(history_path=args.history_path, model_path=args.model_path)

    if not args.nosplitted:
        if not args.Dim2D:
            X_train = np.load(f'{args.input}/X_train_{args.encoding}.npy')
            X_train = X_train.reshape(-1, X_train.shape[1], 1)
            y_train = np.load(f'{args.input}/y_train_{args.encoding}.npy')
            X_val = np.load(f'{args.input}/X_val_{args.encoding}.npy')
            X_val = X_val.reshape(-1, X_val.shape[1], 1)
            y_val = np.load(f'{args.input}/y_val_{args.encoding}.npy')
        else:
            X_train = np.load(f'{args.input}/X_train_{args.encoding}.npy')
            y_train = np.load(f'{args.input}/y_train_{args.encoding}.npy')
            X_val = np.load(f'{args.input}/X_val_{args.encoding}.npy')
            y_val = np.load(f'{args.input}/y_val_{args.encoding}.npy')
    else:
        # Esta seccion esta a medio hacer (pero funcionando)
        # Falta que reciba parametros como test_size, response_col, etc etc
        df = pd.read_csv("{}/{}.csv".format(args.input, args.dataset)) 
        labels = df["activity"]
        data = np.load("{}/{}/{}.npy".format(args.input, args.dataset, 
                                               args.encoding))
        X_train, X_val, y_train, y_val = train_test_split(data, labels, 
                                                          test_size=0.2, random_state=int(args.seed))

    input_shape = X_train.shape[1:]
    
    if not args.Dim2D:
        model = create_fully_connected(input_shape, "binary")
    else:
        model = create_conv1d_bilstm(input_shape, "binary")
    
    history = model.fit(X_train, y_train, validation_data = (X_val, y_val),
                        batch_size=20, epochs=100, callbacks=callbacks_list)
    #plot_results(history)


    val_loss, val_acc = model.evaluate(X_val, y_val)

    if not args.Dim2D:
        metrics_dict = {
            'data_path': args.input,
            'filename': args.dataset if args.nosplitted else "",
            'encoding': args.encoding.replace("_reduced",""),
            'dimension': '1D',
            'seed': args.seed,
            'accuracy': [val_acc],
            'loss': [val_loss]
        }
    else:
        metrics_dict = {
            'data_path': args.input,
            'filename': args.dataset if args.nosplitted else "",
            'encoding': args.encoding,
            'dimension': '2D',
            'seed': args.seed,
            'accuracy': [val_acc],
            'loss': [val_loss]
        }

    metrics_dict.update(get_metrics(model, X_train, y_train, "train_"))
    # Predict2 con x_val 20%
    metrics_dict.update(get_metrics(model, X_val, y_val, "val_"))


    metrics_df = pd.DataFrame(metrics_dict)

    # Export metrics to CSV
    if os.path.exists(args.metrics_path):
        df = pd.read_csv(args.metrics_path)
        metrics_df = pd.concat([df, metrics_df])
    metrics_df.to_csv(args.metrics_path, index=False)
    #plot_results(history)
