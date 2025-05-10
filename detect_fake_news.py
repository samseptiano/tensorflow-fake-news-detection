import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import json
import os

MAX_LENGTH = 500
VOCAB_SIZE = 10000
USER_DATA_PATH = "user_training_data.csv"
FAKE_CSV_PATH = "data/Fake.csv"

def train_model(extra_texts=[], extra_labels=[]):
    # Load base datasets
    df_fake = pd.read_csv(FAKE_CSV_PATH)
    df_fake['label'] = 0

    df_true = pd.read_csv("data/True.csv")
    df_true['label'] = 1

    df = pd.concat([df_fake[['text', 'label']], df_true[['text', 'label']]])

    # Load and include user data if available
    if os.path.exists(USER_DATA_PATH):
        user_df = pd.read_csv(USER_DATA_PATH)
        df = pd.concat([df, user_df], ignore_index=True)

    # Append new user data if provided
    if extra_texts and extra_labels:
        new_data = pd.DataFrame({"text": extra_texts, "label": extra_labels})
        df = pd.concat([df, new_data], ignore_index=True)

        # Append new data to Fake.csv if it is labeled as fake news
        fake_data = new_data[new_data['label'] == 0]  # Only append if label is fake
        if not fake_data.empty:
            fake_data.to_csv(FAKE_CSV_PATH, mode='a', header=False, index=False)

        # Save new data to user_training_data.csv
        new_data.to_csv(USER_DATA_PATH, mode='a', header=not os.path.exists(USER_DATA_PATH), index=False)

    df.dropna(inplace=True)

    # Tokenize and train model
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
    X_train, X_test, y_train, y_test = train_test_split(padded, df['label'], test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 16, input_length=MAX_LENGTH),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    model.save('fake_or_true_news_model.keras')
    with open("tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())

    print("✅ Model trained and saved successfully.")
