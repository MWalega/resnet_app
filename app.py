import os
from flask import Flask, render_template, request
import pickle
from PIL import Image
from torchvision.transforms import transforms
import torch


app = Flask(__name__)
UPLOAD_FOLDER = 'C:/Users/mwale/PycharmProjects/resnet_app/static'
DEVICE = 'cuda'
MODEL = None


def get_class(pred):
    classes = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    return classes[torch.argmax(pred).item()]

def predict(image_path, model):
    # open and transform image from path
    img = Image.open(image_path)
    img = transforms.ToTensor()(img).unsqueeze(0)
    img = transforms.Resize(320)(img)
    img = img.to(DEVICE)

    # predict
    pred = model(img)

    # get correct class
    ans = get_class(pred)

    return ans

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location, MODEL)
            return render_template('index.html', prediction=pred, image_name=image_file.filename)
    return render_template('index.html', prediction=0, image_name=None)

if __name__ == '__main__':
    pickle_in = open('resnet18_pretrained.pkl', 'rb')
    MODEL = pickle.load(pickle_in)
    MODEL.to(DEVICE)
    app.run(debug=True)