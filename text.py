import tensorflow as tf
from tensorflow import keras

print("HHHH",tf.version.VERSION)
myMo=r"C:\Users\winsweb\personal_project\python\ml_servers\heart_diseases_detections\lungs_model.h5"
my_model = tf.keras.models.load_model(myMo)
# print(my_model.summary())



# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)