import numpy as np
import os
import utils
import argparse as arg
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pandas import read_csv, concat, DataFrame
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Embedding, Dense, Dropout, Reshape, Input, concatenate, multiply
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


parser = arg.ArgumentParser(description='Parser for training/evaluationg emoji2vec model')

# Model parameters
parser.add_argument('-d', '--dim', default=300, type=int, help='train a 300 x k projection matrix (embeddings)')
parser.add_argument('-b', '--batch', default=8, type=int, help='size of the mini-batch')
parser.add_argument('-e', '--epochs', default=80, type=int, help='number of training epochs')
parser.add_argument('-dr', '--dropout', default=0.3, type=float, help='amount of dropout to use')
parser.add_argument('-lr', '--learning', default=0.001, type=float, help='learning rate')
parser.add_argument('-u', '--dense', default=600, type=int, help='dense units')
parser.add_argument('-l', '--lstm', default=False, type=bool, help='either use original or an LSTM architecture')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length')

args = parser.parse_args()

embedding_dim = args.dim
batch_size = args.batch
epochs = args.epochs
dropout = args.dropout
learning_rate = args.learning
dense_units = args.dense
seq_length = args.seq
use_lstm = args.lstm

utils.print_settings(embedding_dim, epochs, batch_size, dropout, learning_rate, dense_units, seq_length, use_lstm)

path = os.getcwd()[:os.getcwd().rfind("/")]
emoji_positive = path + "/res/emoji_positive_samples.txt"
emoji_negative = path + "/res/emoji_negative_samples.txt"
emoji_freq = path + "/res/emoji_frequencies.txt"
emoji2vec_visualization = path + "/models/emoji_emb_viz_%dd.csv" % embedding_dim
emoji2vec_weights = path + "/models/weights_%dd.h5" % embedding_dim
emoji2vec_embeddings = path + "/models/emoji_embeddings_%dd.txt" % embedding_dim
glove_filename = path + "/res/glove/" + "glove.6B.%dd.txt" % embedding_dim


# Visualize the TSNE representation of the emoji embeddings
def visualize_emoji_embeddings(top=300):
    # Get the most popular emojis and only plot those
    popular_emojis = [line.split()[0] for line in utils.load_file(emoji_freq)][:top]
    try:
        df = read_csv(emoji2vec_visualization)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Get the data you want ot plot
        x_values = []
        y_values = []
        for index, row in df.iterrows():
            if row["emoji"] in popular_emojis:
                x_values.append(row["x"])
                y_values.append(row["y"])
                ax.text(row["x"], row["y"], row["emoji"], fontname="symbola")
        plt.scatter(x_values, y_values, marker="o", alpha=0.0)
        plt.title("t-SNE visualization of %dd emoji embeddings" % embedding_dim)
        plt.grid()
        plt.savefig(path + "/plots/emoji_%dd_vis.png" % embedding_dim)
    except IOError:
        print("Visualization file not found. Train the emoji embeddings before visualizing them "
              "(they will be automatically saved to %s)" % emoji2vec_visualization)


# Add up the embeddings of the word sequences in the descriptions of the emojis
def sum_emb(word_sequences, embedding_matrix):
    summed_emb = []
    for seq in word_sequences:
        seq_emb = np.zeros(embedding_dim)
        for word_index in seq:
            seq_emb += embedding_matrix[word_index]
        summed_emb.append(seq_emb)
    return np.array(summed_emb)


def emoji2vec_model(emoji_vocab_size):
    emoji_input = Input(shape=(1,), dtype='int32', name='emoji_input')
    emoji_emb = Embedding(emoji_vocab_size, embedding_dim, input_length=1, trainable=True, name='emoji_emb')(emoji_input)
    emoji_emb = Reshape((embedding_dim, ))(emoji_emb)
    word_input = Input(shape=(embedding_dim,), name='word_input')
    x = multiply([emoji_emb, word_input])
    x = Dense(dense_units, activation='tanh')(x)
    x = Dropout(dropout)(x)
    model_output = Dense(2, activation='sigmoid', name='model_output')(x)
    model = Model(inputs=[emoji_input, word_input], outputs=[model_output])
    return model


def emoji2vec_lstm_model(embedding_matrix, emoji_vocab_size, word_vocab_size, sequence_length):
    emoji_input = Input(shape=(1,), dtype='int32', name='emoji_input')
    emoji_emb = Embedding(emoji_vocab_size, embedding_dim, input_length=1, trainable=True, name='emoji_emb')(emoji_input)
    emoji_emb = Reshape((embedding_dim, ))(emoji_emb)

    word_input = Input((sequence_length,), dtype='int32', name='word_input')
    word_emb = Embedding(word_vocab_size, embedding_dim, weights=[embedding_matrix],
                         input_length=sequence_length, trainable=False, name='word_emb')(word_input)
    word_lstm = LSTM(embedding_dim, dropout=dropout, name='word_lstm')(word_emb)

    x = concatenate([emoji_emb, word_lstm])
    x = Dense(dense_units, activation='tanh')(x)
    x = Dropout(dropout)(x)
    model_output = Dense(2, activation='sigmoid', name='model_output')(x)
    model = Model(inputs=[emoji_input, word_input], outputs=[model_output])
    return model


# Solely based on emoji descriptions, obtain the emoji2vec representations for all possible emojis
def train_emoji2vec():
    # Load the true emoji data
    pos_emojis = read_csv(emoji_positive, sep="\t", engine="python", encoding="utf_8", names=["description", "emoji"])
    pos_emojis["label"] = 0

    # Load the false emoji data (negative examples)
    neg_emojis = read_csv(emoji_negative, sep="\t", engine="python", encoding="utf_8", names=["description", "emoji"])
    neg_emojis["label"] = 1

    print("There are %d true emoji descriptions and %d false emoji descriptions." % (len(pos_emojis), len(neg_emojis)))

    # Group all the positive emoji examples by their description
    emoji_grouping = pos_emojis.groupby("emoji")["description"].apply(lambda x: ", ".join(x))
    grouped_by_description = DataFrame({"emoji": emoji_grouping.index, "description": emoji_grouping.values})

    # Build an emoji vocabulary and map each emoji to an index (beginning from 1)
    emojis = grouped_by_description["emoji"].values
    emoji_to_index = {emoji: index + 1 for emoji, index in zip(emojis, range(len(emojis)))}
    emoji_vocab_size = len(emoji_to_index) + 1
    print("There are %d unique emojis." % (emoji_vocab_size - 1))

    # Concatenate and shuffle negative and positive examples of emojis
    all_emojis = concat([pos_emojis, neg_emojis]).sample(frac=1, random_state=150493)

    # Build a word vocabulary and map each emoji to an index (beginning from 1)
    descriptions = all_emojis["description"].values
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(descriptions.tolist())
    word_sequences = tokenizer.texts_to_sequences(descriptions.tolist())
    word_to_index = tokenizer.word_index
    word_vocab_size = len(word_to_index) + 1
    print("There are %d unique words in the descriptions." % (word_vocab_size - 1))

    # Load GLoVe word embeddings
    word_emb = utils.load_vectors(glove_filename)

    # Prepare the word-embedding matrix
    embedding_matrix = utils.get_embedding_matrix(word_emb, word_to_index, embedding_dim, init_unk=False)

    # Prepare training data
    train_emoji = np.array([emoji_to_index[e] for e in all_emojis["emoji"].values])
    print("The emoji tensor shape is ", train_emoji.shape)

    if use_lstm:
        train_words = pad_sequences(word_sequences, maxlen=seq_length, padding='post', truncating='post', value=0.)
    else:
        train_words = sum_emb(word_sequences, embedding_matrix)
    print("The descriptions tensor shape is ", train_words.shape)

    labels = to_categorical(np.asarray([label for label in all_emojis["label"].values]))
    print("The label tensor shape is ", labels.shape)

    # Build the emoji DNN model
    if use_lstm:
        model = emoji2vec_lstm_model(embedding_matrix, emoji_vocab_size, word_vocab_size, seq_length)
    else:
        model = emoji2vec_model(emoji_vocab_size)
    my_optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.99, decay=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=my_optimizer, metrics=["categorical_accuracy", utils.f1_score])
    print(model.summary())

    plot_model(model, to_file=path + '/plots/emoji2vec_' + str(embedding_dim) + 'd_model_summary.png',
               show_shapes=False, show_layer_names=True)

    # Prepare the callbacks and fit the model
    save_best = ModelCheckpoint(monitor='val_categorical_accuracy', save_best_only=True, filepath=emoji2vec_weights)
    reduceLR = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=15, verbose=1)
    history = model.fit([train_emoji, train_words], labels, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, verbose=1, callbacks=[save_best, reduceLR, early_stopping])

    # Plot accuracy and loss
    utils.plot_training_statistics(history, "/plots/emoji2vec_%dd" % embedding_dim,
                                   plot_validation=True, acc_mode="categorical_accuracy", loss_mode="loss")

    # Get the weights of the trained emoji model
    weights = [layer.get_weights()[0] for layer in model.layers if layer.name == 'emoji_emb']
    weights = weights[0]

    # Get the emoji embeddings and save them to file
    embeddings = DataFrame(weights[1:])
    embeddings = concat([grouped_by_description["emoji"], embeddings], axis=1)
    embeddings.to_csv(emoji2vec_embeddings, sep=" ", header=False, index=False)

    # Get the t-SNE representation
    tsne = TSNE(n_components=2, perplexity=30, init="pca", n_iter=5000)
    trans = tsne.fit_transform(weights)

    # Save the obtained emoji visualization
    visualization = DataFrame(trans[1:], columns=["x", "y"])
    visualization["emoji"] = grouped_by_description["emoji"].values
    visualization.to_csv(emoji2vec_visualization)

    # Visualize the embeddings as a t-sne figure
    visualization.plot("x", "y", kind="scatter", grid=True)
    plt.savefig(path + "/plots/tsne_%dd.png" % embedding_dim)


def analogy_task():
    try:
        emoji2vec_str = utils.load_vectors(filename=emoji2vec_embeddings)
        # Convert to unicode all emoji entries in the dictionary of emoji embeddings
        emoji2vec = {}
        for k, v in emoji2vec_str.items():
            unicode_emoji = utils.convert_emoji_to_unicode(k)
            emoji2vec[unicode_emoji] = v
        # Get some intuition whether the model is good by seeing what analogies it can make based on what it learnt
        utils.make_analogy("👑", "🚹", "🚺", emoji2vec)  # Crown - Man + Woman
        utils.make_analogy("👑", "👦", "👧", emoji2vec)  # Crown - Boy + Girl
        utils.make_analogy("💵", "🇺🇸", "🇬🇧", emoji2vec)
        utils.make_analogy("💵", "🇺🇸", "🇪🇺", emoji2vec)
        utils.make_analogy("👪", "👦", "👧", emoji2vec)
        utils.make_analogy("🕶", "☀️", "⛈", emoji2vec)  # Sunglasses - Sun + Cloud
        utils.make_analogy("☂", "⛈️", "☀", emoji2vec)  # Umbrella - Clouds + Sun
        utils.make_analogy("🍣", "🏯️", "🏰", emoji2vec)  # Sushi - Japanese Castle + European Castle
        utils.make_analogy("👹", "🏯️", "🏰", emoji2vec)  # Japanese Ogre - Japanese Castle + European Castle
        utils.make_analogy("🍣", "🗼️", "🗽", emoji2vec)  # Sushi - Japanese Tower + Statue of Liberty
        utils.make_analogy("🍣", "🗾️", "🗽", emoji2vec)  # Sushi - Japanese Tower + Statue of Liberty
        utils.make_analogy("🍣", "🏯️", "🗽", emoji2vec)  # Sushi - Japanese Castle + Statue of Liberty
        utils.make_analogy("🐅", "🐈️", "🐕", emoji2vec)  # Jaguar - Cat + Dog
        utils.make_analogy("🐆", "🐈️", "🐕", emoji2vec)  # Leopard - Cat + Dog
        utils.make_analogy("🐭", "🐈️", "🐕", emoji2vec)  # Mouse - Cat + Dog
        utils.make_analogy("🌅", "🌞️", "🌙", emoji2vec)  # Sunrise - Sun + Moon
        utils.make_analogy("🌅", "🌞️", "🌑", emoji2vec)  # Sunrise - Sun + Moon
        utils.make_analogy("🌃", "🌙️", "🌞", emoji2vec)  # Night with stars - Moon + Sun With Face
        utils.make_analogy("🌃", "🌑️", "☀", emoji2vec)  # Night with stars - Moon + Sun With Face
        utils.make_analogy("🌃", "🌙️️", "☀", emoji2vec)  # Night with stars - Moon + Sun With Face
        utils.make_analogy("😴", "💤️", "🏃", emoji2vec)  # Sleeping face - sleeping symbol + running
        utils.make_analogy("😴", "🛌️", "🏃", emoji2vec)  # Sleeping face - sleeping accommodatin + running
        utils.make_analogy("😴", "🛏", "🏃", emoji2vec)  # Sleeping face - bed + active symbol (running)
        utils.make_analogy("🏦", "💰", "🏫", emoji2vec)  # Money - Bank + School
        utils.make_analogy("🏦", "💰", "🏥", emoji2vec)  # Money - Bank + Hospital
        utils.make_analogy("💉", "🏥", "🏦", emoji2vec)  # Syringe - Hospital + Bank
        utils.make_analogy("💊", "🏥", "🏦", emoji2vec)  # Pill - Hospital + Bank
        utils.make_analogy("💒", "💍", "👰", emoji2vec)  # Wedding - Ring + Bride
        utils.make_analogy("💒", "💑", "💔", emoji2vec)  # Wedding - Couple + Broken Heart
        utils.make_analogy("💒", "❤", "💔", emoji2vec)  # Wedding - Heart + Broken Heart
        utils.make_analogy("😀", "💰", "🤑", emoji2vec)  # Grinning person - Money + Money Face
        utils.make_analogy("😠", "💰", "🤑", emoji2vec)  # Angry person - Money + Money Face
    except IOError:
        print("Emoji embeddings not found at the provided embeddings file %s. "
              "You have to train them before proceeding to make analogies." % emoji2vec_embeddings)


if __name__ == "__main__":
    if not os.path.exists(emoji2vec_embeddings):
        train_emoji2vec()                 # train model (can skip if already have embeddings)
    visualize_emoji_embeddings()
    analogy_task()
