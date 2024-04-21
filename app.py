# app.py

from flask import Flask, render_template, request
import os
from flask_cors import CORS
import time
from flask import send_from_directory
from SigVerAPI.verifier import verifySignature
from SigVerAPI.uploadAndTrain import *

app = Flask(__name__)
cors = CORS(app, origin="*")
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/admin')
def admin():
    return render_template("admin.html")

@app.route('/Static/<path:path>')
def send_report(path):
    return send_from_directory('static', path)


@app.route('/verifySignature',methods=["POST"])
def verify():
    id = request.form['person_id']

    try: image=request.files['image'] 
    except: image=""
    
    if id == "":
        return "<H1> Please enter a valid person ID <H1>"
    if image == "":
        return "<H1> Please upload an image. <H1>"
    
    id_int = int(id)
    if id_int > 14:
        return "<H1> The id doesn't exist. Enter valid ID. <H1>"
    
    print(image.filename)
    image_path=savephoto(image,image.filename)
    print("The passed id is: "+id, "The image path is: "+image_path)
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
    print("File is saved!!")
    return image_path_normal

@app.route('/UploadSignatures',methods=["POST"])
def UploadSignatures():
    genuine_images = request.files.getlist('genuine_image')
    forged_images = request.files.getlist('forged_image')

    # try: image=request.files['image'] 
    # except: image=""
    
    # if image == "":
    #     return "<H1> Please upload images. <H1>"
    
    if upload_train_test_image(forged_images,genuine_images):
        return "<H1> The images are uploaded and CSV files are created <H1>"