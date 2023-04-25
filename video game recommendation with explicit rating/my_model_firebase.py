from tensorflow.keras import layers
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import mysql.connector
import firebase_admin
from firebase_admin import firestore
# from firebase_admin import db
from firebase_admin import credentials

################### Here we retrain the MODEL ####################


def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(
            'mobile-app-543bf-firebase-adminsdk-d937c-f23c606ee2.json')
        firebase_admin.initialize_app(cred)

def retrain():
    initialize_firebase()
    db = firestore.client()
    comments_ref = db.collection('comments')

    # Fetch the snapshot and process the documents
    comments_records = []

    for doc in comments_ref.stream():
        comments_record = {
            'commentDescription': doc.get('commentDescription'),
            'starsNumber': doc.get('starsNumber'),
            'gameId': doc.get('gameId'),
            'userId': doc.get('userId')
        }
        comments_records.append(comments_record)

    all_user_ids = [record['userId'] for record in comments_records]
    all_game_ids = [record['gameId'] for record in comments_records]
    users_ratings = [record['starsNumber'] for record in comments_records]

    print(all_user_ids)
    print(all_game_ids)
    print(users_ratings)
    user_ids = list(set(all_user_ids))

    # PREPROCESSING
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}

    game_ids = list(set(all_game_ids))
    game2game_encoded = {x: i for i, x in enumerate(game_ids)}
    game_encoded2game = {i: x for i, x in enumerate(game_ids)}

    user = list(map(lambda x: user2user_encoded[x], all_user_ids))
    game = list(map(lambda x: game2game_encoded[x], all_game_ids))

    x = [list(pair) for pair in zip(user, game)]
    x = np.array(x, dtype=int)

    num_users = len(user2user_encoded)
    num_games = len(game_encoded2game)

    users_ratings = [np.float32(x) for x in users_ratings]

    min_rating = min(users_ratings)
    max_rating = max(users_ratings)
    y = [(x - min_rating) / (max_rating - min_rating)
         for x in users_ratings]
    y = np.array(y, dtype='float64')

    train_indices = int(0.7*len(x))
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )
    print(x)
    print(y)
    # Close the database connection

    EMBEDDING_SIZE = 50

    class RecommenderNet(keras.Model):
        def __init__(self, num_users, num_games, embedding_size, **kwargs):
            super().__init__(**kwargs)
            self.num_users = num_users
            self.num_games = num_games
            self.embedding_size = embedding_size
            # categorical variable with a high cardinality
            self.user_embedding = layers.Embedding(
                num_users,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.user_bias = layers.Embedding(num_users, 1)
            self.game_embedding = layers.Embedding(
                num_games,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.game_bias = layers.Embedding(num_games, 1)

        def call(self, inputs):
            user_vector = self.user_embedding(inputs[:, 0])
            user_bias = self.user_bias(inputs[:, 0])
            game_vector = self.game_embedding(inputs[:, 1])
            game_bias = self.game_bias(inputs[:, 1])
            dot_user_game = tf.tensordot(user_vector, game_vector, 2)
            # Add all the components (including bias)
            x = dot_user_game + user_bias + game_bias
            # The sigmoid activation forces the rating to between 0 and 1
            return tf.nn.sigmoid(x)

    model = RecommenderNet(num_users, num_games, EMBEDDING_SIZE)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
        # ,  metrics=['accuracy', f1_m, precision_m, recall_m]
    )
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=20,
        verbose=1,
        validation_data=(x_val, y_val),
    )
    model.save('my_model_updated')


def recommend(user_id):
    initialize_firebase()

    db = firestore.client()
    comments_ref = db.collection('comments')

    # Fetch the snapshot and process the documents
    comments_records = []

    for doc in comments_ref.stream():
        comments_record = {
            'commentDescription': doc.get('commentDescription'),
            'starsNumber': doc.get('starsNumber'),
            'gameId': doc.get('gameId'),
            'userId': doc.get('userId')
        }
        comments_records.append(comments_record)
    all_user_ids = [record['userId'] for record in comments_records]
    all_game_ids = [record['gameId'] for record in comments_records]
    users_ratings = [record['starsNumber'] for record in comments_records]

    # PREPROCESSING
    user_ids = list(set(all_user_ids))
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}

    game_ids = list(set(all_game_ids))
    game2game_encoded = {x: i for i, x in enumerate(game_ids)}
    game_encoded2game = {i: x for i, x in enumerate(game_ids)}
    data = {
        'user_id': all_user_ids,
        'game_id': all_game_ids,
        'user_rating': users_ratings
    }

    df = pd.DataFrame(data)
    games_df = df["game_id"]
    games_df = games_df.rename('id')
    games_df = games_df.to_frame()
    games_df = games_df.drop_duplicates()

    games_played_by_user = df[df.user_id == user_id]

    games_not_played = games_df[
        ~games_df["id"].isin(games_played_by_user.game_id.values)
    ]["id"]

    games_not_played = list(
        set(games_not_played).intersection(set(game2game_encoded.keys()))
    )
    games_not_played = [[game2game_encoded.get(x)] for x in games_not_played]

    user_encoder = user2user_encoded.get(user_id)

    user_game_array = np.hstack(
        ([[user_encoder]] * len(games_not_played), games_not_played)
    )
    model = tf.keras.models.load_model("my_model_updated")

    ratings = model.predict(user_game_array).flatten()
    print((user_game_array))
    print((ratings))

    # Extract the game indices from the user_game_array
    game_indices = user_game_array[:, 1]

    # Create a dictionary mapping game index to rating value
    game_ratings = {game_index: rating for game_index,
                    rating in zip(game_indices, ratings)}
    # Replace game index with real game ID in the game_ratings dictionary
    game_ratings_with_ids = {
        game_encoded2game[game_index]: rating for game_index, rating in game_ratings.items()}

    min_value = min(game_ratings_with_ids.values())
    max_value = max(game_ratings_with_ids.values())

# Scale the values and create a new dictionary with the original keys and scaled values
    scaled_game_ratings_with_ids = {key: int(round(((value - min_value) / (max_value - min_value)
                                                    * (1 - 0) + 0), 2) * 100) for key, value in game_ratings_with_ids.items()}

    return scaled_game_ratings_with_ids


#retrain()
print("recommendation ",recommend("CgVbyyHmer3HUgDleQFW"))
