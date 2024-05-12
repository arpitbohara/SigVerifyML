from SigVerAPI.verifier import verifySignature
import os
import dotenv
dotenv.load_dotenv()
from SigVerAPI.fileOperations import makeCSV

def savephoto(image,image_name,dir,message)->str:
    if not os.path.exists(dir): os.mkdir(dir)
    image_path = os.path.abspath(f"{dir}/{image_name}")
    image_path_normal=os.path.normpath(image_path)
    image.save(image_path)
    print(message+" Image is saved!!")
    return image_path_normal

def get_new_trained_id():
    test_feature_folder=os.getenv("TESTING_FEATURE_FOLDER")
    test_files=os.listdir(test_feature_folder)
    start=len(test_files)+1
    return start

def makeCSVsingle():
    forged_image_paths=os.getenv("FORGED_IMAGE_PATH")
    genuine_image_paths=os.getenv("GENIUNE_IMAGE_PATH")
    start=get_new_trained_id()
    end=start+1
    makeCSV(genuine_image_paths,forged_image_paths,start,end)

def upload_train_test_image(forged_image,geniuine_image):
    new_person_id=get_new_trained_id()
    forged_image_paths=os.getenv("FORGED_IMAGE_PATH")
    genuine_image_paths=os.getenv("GENIUNE_IMAGE_PATH")
    if len(str(new_person_id))==2: new_person_id="0"+str(new_person_id)
    if len(str(new_person_id))==1: new_person_id="00"+str(new_person_id)
    counter=0
    for image in forged_image:
        image_name='021'+new_person_id+"_00"+str(counter)+".png"
        savephoto(image,image_name,dir=forged_image_paths, message = "Forged")
        counter=counter+1
    counter=0
    for image in geniuine_image:
        image_name=new_person_id+new_person_id+"_00"+str(counter)+".png"
        savephoto(image,image_name,dir=genuine_image_paths,  message = "Genuine")
        counter=counter+1
    makeCSVsingle()
    return True