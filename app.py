import gradio as gr
import pickle
import tensorflow as tf
from tensorflow import keras
# data = pickle.load(open('train_data.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))
model = tf.keras.models.load_model('my_model')
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def predict_input_image(img):
    img_4d = img.reshape(-1,180,180,3)
    predic=model.predict(img_4d)[0]
    return {class_names[i]:float(predic[i]) for i in range(5)}

image = gr.inputs.Image(shape=(180,180))
label = gr.outputs.Label(num_top_classes=5)
gr.Interface(fn=predict_input_image,inputs=image,outputs=label,interpretation='default').launch(debug=True,share=True)