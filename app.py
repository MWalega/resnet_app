import os
from flask import Flask, render_template, request
from PIL import Image
from torchvision.transforms import transforms
import torchvision.models as models
import torch
import torch.nn.functional


app = Flask(__name__)
UPLOAD_FOLDER = 'C:/Users/mwale/PycharmProjects/resnet_app/static'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None


def get_class(pred):
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Show top categories per image
    top3_prob, top3_catid = torch.topk(pred, 3)
    top_3 = []
    print(torch.max(pred))
    for i in range(top3_prob.size(0)):
        top_3.append((categories[top3_catid[i]], "{:.2%}".format(top3_prob[i].item())))

    return top_3

def predict(image_path, model):
    # open and transform image from path
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = preprocess(img)
    img = img.unsqueeze(0)
    img = img.to(DEVICE)

    # predict
    with torch.no_grad():
        out = model(img)

    out = torch.nn.functional.softmax(out[0], dim=0)

    # get correct class
    ans = get_class(out)

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
            preds = predict(image_location, MODEL)
            return render_template('index.html', prediction=preds, image_name=image_file.filename)
    return render_template('index.html', prediction=0, image_name=None)

if __name__ == '__main__':
    MODEL = models.resnet18(pretrained=True)
    MODEL.eval()
    MODEL.to(DEVICE)
    app.run(debug=True)