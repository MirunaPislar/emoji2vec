import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import keras.backend as K
from numpy.random import seed
path = os.getcwd()[:os.getcwd().rfind('/')]


def load_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text.split("\n")


def save_file(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# Load a set of pre-trained embeddings (can be GLoVe or emoji2vec)
def load_vectors(filename):
    print("\nLoading vector mappings from %s..." % filename)
    word2vec_map = {}
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        weights = np.asarray(values[1:], dtype='float32')
        word2vec_map[word] = weights
    f.close()
    print('Found %s word vectors and with embedding dimmension %s'
          % (len(word2vec_map), next(iter(word2vec_map.values())).shape[0]))
    return word2vec_map


# Compute the word-embedding matrix
def get_embedding_matrix(word2vec_map, word_to_index, embedding_dim, init_unk=True, variance=None):
    # Get the variance of the embedding map
    if init_unk and variance is None:
        variance = embedding_variance(word2vec_map)
        print("Word vectors have variance ", variance)
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors)
    embedding_matrix = np.zeros((len(word_to_index) + 1, embedding_dim))
    for word, i in word_to_index.items():
        embedding_vector = word2vec_map.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        elif init_unk:
            # Unknown tokens are initialized randomly by sampling from a uniform distribution [-var, var]
            seed(1337603)
            embedding_matrix[i] = np.random.uniform(-variance, variance, size=(1, embedding_dim))
        # else:
        #    print("Not found: ", word)
    return embedding_matrix


# Calculate the variance of an embedding (like glove, word2vec, emoji2vec, etc)
# Used to sample new uniform distributions of vectors in the interval [-variance, variance]
def embedding_variance(vec_map):
    variance = np.sum([np.var(vec) for vec in vec_map.values()]) / len(vec_map)
    return variance


# Compute the similarity of 2 vectors, both of shape (n, )
def cosine_similarity(u, v):
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(u ** 2))
    norm_v = np.sqrt(np.sum(v ** 2))
    cosine_distance = dot / (norm_u * norm_v)
    return cosine_distance


# Convert emojis to unicode representations by removing any variation selectors
# Info: http://www.unicode.org/charts/PDF/UFE00.pdf
def convert_emoji_to_unicode(emoji):
    unicode_emoji = emoji.encode('unicode-escape')
    find1 = unicode_emoji.find(b"\\ufe0f")
    unicode_emoji = unicode_emoji[:find1] if find1 != -1 else unicode_emoji
    find2 = unicode_emoji.find(b"\\ufe0e")
    unicode_emoji = unicode_emoji[:find2] if find2 != -1 else unicode_emoji
    return unicode_emoji


# Performs the word analogy task: a is to b as c is to ____.
def make_analogy(a, b, c, vec_map):
    a = convert_emoji_to_unicode(a)
    b = convert_emoji_to_unicode(b)
    c = convert_emoji_to_unicode(c)

    e_a, e_b, e_c = vec_map[a], vec_map[b], vec_map[c]

    best_list = {}
    for v in vec_map.keys():
        if v in [a, b, c]:      # best match shouldn't be one of the inputs, so pass on these
            continue
        best_list[v] = cosine_similarity(e_b - e_a, vec_map[v] - e_c)

    sorted_keys = sorted(best_list, key=best_list.get, reverse=True)[:5]
    print(str.format('{} - {} + {} = {}', a.decode('unicode-escape'), b.decode('unicode-escape'),
                     c.decode('unicode-escape'), [r.decode('unicode-escape') for r in sorted_keys]))
    print()


# Custom metric function adjusted from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def f1_score(y_true, y_pred):
    # Recall metric. Only computes a batch-wise average of recall,
    # a metric for multi-label classification of how many relevant items are selected.
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    # Precision metric. Only computes a batch-wise average of precision,
    # a metric for multi-label classification of how many selected items are relevant.
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    f_score = 2 * ((precision*recall) / (precision+recall))
    return f_score


# Plot the accuracy and loss for the model trained
def plot_training_statistics(history, plot_name, plot_validation=False, acc_mode='acc', loss_mode='loss'):
    # Plot Accuracy
    plt.figure()
    plt.plot(history.history[acc_mode], 'k-', label='Training Accuracy')
    if plot_validation:
        plt.plot(history.history['val_' + acc_mode], 'r--', label='Validation Accuracy')
        plt.title('Training vs Validation Accuracy')
    else:
        plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='center right')
    plt.ylim([0.0, 1.0])
    plt.savefig(path + plot_name + "_acc.png")
    print("Plot for accuracy saved to %s" % (path + plot_name + "_acc.png"))

    # Plot Loss
    plt.figure()
    plt.plot(history.history[loss_mode], 'k-', label='Training Loss')
    if plot_validation:
        plt.plot(history.history['val_' + loss_mode], 'r--', label='Validation Loss')
        plt.title('Training vs Validation Loss')
    else:
        plt.title('Training Loss ')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='center right')
    plt.savefig(path + plot_name + "_loss.png")
    print("Plot for loss saved to %s" % (path + plot_name + "_loss.png"))


# Plot the Receiver Operating Characteristics
def plot_roc(model, all_emojis, train_emoji, train_words):
    y = []
    y_pred = []
    y_test = [l for l in all_emojis['label'].values]
    prediction_probability = model.predict([train_emoji, train_words])

    for i, (_) in enumerate(prediction_probability):
        predicted = np.argmax(prediction_probability[i])
        y.append(int(y_test[i]))
        y_pred.append(predicted)

    fpr, tpr, threshold = metrics.roc_curve(y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.title('Receiver Operating Characteristic (ROC) Curve for learned emoji')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot(fpr, tpr, 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid()
    plt.show()
    plt.savefig(path + "/plots/roc.png")


# Method that prints the settings for each DNN model
def print_settings(embedding_dim, epochs, batch_size, dropout, learning_rate, dense_units, seq_length, use_lstm):
    print("==================================================================\n")
    print('{:>25}'.format("Hyperparameters"))
    print("==================================================================\n")
    print('{:>25}  {:>10}'.format("Embedding dimension", embedding_dim))
    print('{:>25}  {:>10}'.format("Epochs", epochs))
    print('{:>25}  {:>10}'.format("Batch size", batch_size))
    print('{:>25}  {:>10}'.format("Dropout", dropout))
    print('{:>25}  {:>10}'.format("Learning rate", learning_rate))
    print('{:>25}  {:>10}'.format("Dense units", dense_units))
    print('{:>25}  {:>10}'.format("Use LSTM", use_lstm))
    if use_lstm:
        print('{:>25}  {:>10}'.format("Seq length", seq_length))
    print("==================================================================\n")
