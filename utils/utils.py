import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import numpy as np

def acc_f1(y_prediction, y_true):
    y_prediction = y_prediction.numpy()
    y_true = y_true.numpy()
    f1 = f1_score(y_true, y_prediction, average="macro")
    correct = np.sum((y_true == y_prediction).astype(int))
    acc = correct / y_prediction.shape[0]
    return acc, f1


def class_report(y_prediction, y_true):
    y_true = y_true.numpy()
    y_prediction = y_prediction.numpy()
    classify_report = classification_report(y_true, y_prediction)
    print('\n\nclassify_report:\n', classify_report)


# 无图形界面需要加，否则plt报错
plt.switch_backend('agg')


def loss_acc_plot(history, save_path):
    train_loss = history['train_loss']
    eval_loss = history['eval_loss']
    train_accuracy = history['train_acc']
    eval_accuracy = history['eval_acc']

    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(2, 1, 1)
    plt.title('loss during train')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    epochs = range(1, len(train_loss)+1)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, eval_loss)
    plt.legend(['train_loss', 'eval_loss'])

    fig.add_subplot(2, 1, 2)
    plt.title('accuracy during train')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_accuracy)
    plt.plot(epochs, eval_accuracy)
    plt.legend(['train_acc', 'eval_acc'])

    plt.savefig(save_path)
