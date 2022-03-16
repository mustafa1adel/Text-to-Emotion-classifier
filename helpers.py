import re
import pandas as pd
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks

def clean_text(text):
    url = re.compile(r"\S*https?:\S*")
    mentions = re.compile(r"@[a-zA-Z0-9_]+")
    special_ch = re.compile('[-\+!~@#$%^_&*()={}\[\]:;<.>?/\'"]')
    new_lines = re.compile('\s+')
    numbers = re.compile('[0-9]')
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    text = url.sub("", text)
    text = mentions.sub("", text)
    text = special_ch.sub("", text)
    text = new_lines.sub(" ", text)
    text = numbers.sub("", text)
    text = emoji_pattern.sub("", text)

    return text.strip()

def check_duplicates(df):
    return bool(df.duplicated().sum())

def check_empty(df):
    return df.isnull().values.any() or df.isna().values.any()

# load a clean dataset
def load_dataset(filename):
    return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

# fit an encoder
def create_encoder(labels):
    encoder = LabelEncoder()
    return encoder.fit(labels)

# encode a list of labels
def encode_label(encoder, labels):
    return encoder.transform(labels)


# define the model
def define_model(length, vocab_size):
    callback = callbacks.EarlyStopping(monitor='val_loss', patience=1)
    # Model Architecture
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length= length))
    model.add(Conv1D(filters = 32, kernel_size= 4, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Conv1D(filters = 64, kernel_size= 8, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Conv1D(filters = 64, kernel_size= 2, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation = 'softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    # summarize
    print(model.summary())

    return model, callback
