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
query = "SELECT * FROM `user_preferences`;"
cursor.execute(query)

input_data = []

result = cursor.fetchall()

print(result)


# Close the database connection
cursor.close()
cnx.close()


model = tf.keras.models.load_model("my_model")

'''
# Convert input data to a NumPy array and normalize if necessary
input_data = np.array(input_data)
# input_data = input_data / some_value  # Uncomment and replace 'some_value' if you need to normalize the data

# Make predictions using the model
predictions = model.predict(input_data)

# Do something with the predictions, e.g., print them
print(predictions)
'''
