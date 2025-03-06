import gradio as gr
import numpy as np
import cv2
from tensorflow import keras

model = keras.models.load_model("cats_vs_dogs_transfer_learning.keras")

def preprocess_image(image):
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    image = preprocess_image(image)
    prediction = model.predict(image)[0][0]
    return "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Cats vs Dogs Classifier 🐱🐶",
    description="Upload an image of a cat or dog, and the model will classify it!"
)

if __name__ == "__main__":
    iface.launch()


#  http://127.0.0.1:7860 to access the GUI for it