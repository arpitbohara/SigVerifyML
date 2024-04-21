
# How Verification Works

<b>This Repository contains code for verification of signature</b>

<br/>Tensorflow is used and details of folder is as follows -
<br/>* app.py is the main flask file for the application
<br/>* static folder contains css and js files 
<br/>* templates folder contains html files
<br/>* SigVerAPI folder contains all the apis.
<br/>* sigverifyenv is the environment files for the project
<br/>* forged and real folders contain dataset for this project in form of real and forged images respectively
<br/>* Data\Features\Training folder will save the trained csv
<br/>* Data\Features\Testing folder will save the tested csv

<br/>Images are stored in the format : <b>XXXZZZ_YYY</b>
<br/>XXX denotes id of the person who has signed on the document(ex -001)
<br/>ZZZ denotes the id of the person to whom the sign belongs in actual(ex- 001)
<br/>YYY deontes the n'th no.of attempt

<br/>Now if <b>XXX == ZZZ </b>then image is <b>genuine</b> otherwise the signature is forged


## Instructions to Run the App

1. Activate the environment in the terminal.
    ```bash
    .\Sigverfiyenv\Scripts\activate.bat
    ```
2. Set Flask environment to development:
    ```bash
    $env:FLASK_ENV = "development"
    ```
3. Run Flask:
    ```bash
    flask run
    ```
4. Open a web browser and enter `localhost:5000` for the verification page.
5. For signature upload and training, open `localhost:5000/admin`.
