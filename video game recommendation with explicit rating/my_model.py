import mysql.connector
import numpy as np
import tensorflow as tf


cnx = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="gamer_guide"
)

# Fetch data from the database
cursor = cnx.cursor()
query = "SELECT user_id,game_id,rating FROM `user_preferences`;"
cursor.execute(query)

input_data = []

result = cursor.fetchall()
all_user_ids = [item[0] for item in result]
all_game_ids = [item[1] for item in result]
users_ratings = [item[2] for item in result]
# PREPROCESSING
user_ids = list(set(all_user_ids))
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

game_ids = list(set(all_game_ids))
game2game_encoded = {x: i for i, x in enumerate(game_ids)}
game_encoded2game = {i: x for i, x in enumerate(game_ids)}

user = list(map(lambda x: user2user_encoded[x], all_user_ids))
game = list(map(lambda x: game2game_encoded[x], all_game_ids))

num_users = len(user2user_encoded)
num_games = len(game_encoded2game)


users_ratings = [np.float32(x) for x in users_ratings]
min_rating = min(users_ratings)
max_rating = max(users_ratings)
print(
    "Number of users: {}, Number of Games: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_games, min_rating, max_rating
    )
)

# Close the database connection
cursor.close() 
cnx.close()
model = tf.keras.models.load_model("my_model")
# dataList = [[964,  11],
#             [964, 127],
#             [964, 124],
#             [964,  40],
#             [964,   8],
#             [964,  10],
#             [964,  21],
#             [964,  53],
#             [964,  72],
#             [964,  73],
#             [964,   5],
#             [964,  78],
#             [964,  42],
#             [964, 125],
#             [964, 133],
#             [964, 129],
#             [964, 135],
#             [964,  58],
#             [964, 108],
#             [964, 130],
#             [964,  27],
#             [964, 120],
#             [964, 103],
#             [964, 116],
#             [964,   9],
#             [964, 101],
#             [964, 139],
#             [964,  23],
#             [964,  54],
#             [964,  39],
#             [964,  51],
#             [964,  84],
#             [964,  32],
#             [964,  16],
#             [964,  95],
#             [964,   3],
#             [964,  98],
#             [964, 137],
#             [964, 126],
#             [964,  55],
#             [964, 113],
#             [964,  60],
#             [964,  25],
#             [964,  13],
#             [964,  71],
#             [964,  36],
#             [964,  75],
#             [964,  83],
#             [964,  31],
#             [964,  90],
#             [964,  69],
#             [964, 118],
#             [964,  82],
#             [964, 119],
#             [964,  15],
#             [964,  41],
#             [964,   6],
#             [964, 109],
#             [964,  88],
#             [964,  96],
#             [964, 122],
#             [964,  99],
#             [964,  14],
#             [964,  68],
#             [964,  74],
#             [964,  24],
#             [964,  22],
#             [964,  81],
#             [964,  59],
#             [964,  62],
#             [964,  85],
#             [964, 112],
#             [964, 115],
#             [964,  45],
#             [964,  91],
#             [964,  29],
#             [964,  34],
#             [964,  93],
#             [964, 100],
#             [964,  64],
#             [964, 106],
#             [964,  77],
#             [964, 107],
#             [964,  33],
#             [964,  80],
#             [964,   2],
#             [964,  65],
#             [964,  92],
#             [964,  48],
#             [964,  26],
#             [964, 121],
#             [964,  12],
#             [964,  57],
#             [964,  67],
#             [964,  28],
#             [964, 104],
#             [964,  49],
#             [964, 114],
#             [964, 134],
#             [964,  66],
#             [964,   4],
#             [964,  56],
#             [964, 131],
#             [964,  89],
#             [964,  97],
#             [964, 102],
#             [964, 138],
#             [964,   0],
#             [964,  70],
#             [964, 110],
#             [964, 136],
#             [964, 128],
#             [964,  63],
#             [964,  86],
#             [964,  47],
#             [964,  37],
#             [964,  61],
#             [964,   1],
#             [964, 132],
#             [964,  46],
#             [964,  44],
#             [964,  87],
#             [964, 111],
#             [964, 123],
#             [964,  18],
#             [964, 117],
#             [964,  50],
#             [964,  79],
#             [964,  38],
#             [964,  30],
#             [964,  76],
#             [964, 105],
#             [964,  19],
#             [964,  52],
#             [964,  35],
#             [964,  43],
#             [964,  17],
#             [964,  94]]
# arr = np.array(dataList, dtype=np.int64)


# print(type(arr))

# print(model.summary())
# ratings = model.predict(arr)
# min_value = ratings.min()
# max_value = ratings.max()
# scaled_ratings = (ratings - min_value) / (max_value - min_value) * (1 - 0) + 0
#print(scaled_ratings)
