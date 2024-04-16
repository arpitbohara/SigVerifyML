# app.py

from flask import Flask, render_template, request
import os
from flask_cors import CORS
import time
from flask import send_from_directory
from SigVerAPI.verifier import verifySignature

app = Flask(__name__)
cors = CORS(app, origin="*")
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/Static/<path:path>')
def send_report(path):
    return send_from_directory('static', path)


@app.route('/verifySignature',methods=["POST"])
def verify():
    id = request.form['person_id']
    image= request.files['image']
    print(image.filename)
    image_path=savephoto(image,image.filename)
    print(id,image_path)
    if verifySignature(id,image_path):
        return "<H1> The Image is Genuine <H1>"
    else:
        return "<H1> The Image is Forged!! <H1>"

def savephoto(image,image_name)->str:
    dir="uploads"
    if not os.path.exists(dir): os.mkdir(dir)
    image_path = os.path.abspath(f"{dir}/{image_name}")
    image_path_normal=os.path.normpath(image_path)
    image.save(image_path)
    print("file is saved!!")
    return image_path_normal
