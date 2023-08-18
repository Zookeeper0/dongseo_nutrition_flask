"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import datetime
import io
import os

import torch
from flask import Flask, redirect, render_template, request, send_file
from PIL import Image
from ultralytics import YOLO, utils
import pandas as pd
from flask import jsonify


app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if not file:
            return

        print(file)
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        print(img)
        
        results = model(img)
        
        
        results.render()  # updates results.imgs with boxes and labels
        main_folder=f'static/{now_time}.png'
        
        results.crop(save=True, save_dir=main_folder)  
        
        list =[]
        try:
            for item in os.listdir(f'{main_folder}/crops'):
                print(f'item : {item}')
                list.append({"name": item, "imgUrl": f'{main_folder}/crops/{item}/image0.jpg' })
                
            return jsonify(list)
        except:
            return TypeError("error!!!!")
        
        # img_savename = f"static/{now_time}.png"
        # Image.fromarray(results.ims[0]).save(img_savename)
        
            


    return render_template("index.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('./yolov5/', 'custom', path='./yolov5/runs/train/1301-1800_10000/weights/best.pt', source='local')

    # Load the model
    # model = torch.load('c:/Users/SAMSUNG/OneDrive/dongseo/dongseo_nutrition_flask/what/weights/best.py')

    # Set the model to evaluation mode
    model.eval()
    # model = torch.hub.load('./wpqkf/weights/best.pt', pretrained=True)
    # force_reload = recache latest code
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
