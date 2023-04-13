from flask import Flask,flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf

UPLOAD_FOLDER = 'static/img/'

app = Flask(__name__)
app.secret_key = 'hugomilesi'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['jpeg', 'png', 'jpg'])

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods = ['POST', 'GET'])
def predict():
    if 'file' not in request.files:
        flash('No file Part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No image Selected for prediction')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # file path to save the images
        img_path = 'static/img/' + filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels = 3)
        img = tf.image.resize(img, size = [224, 224])
        img = img/225
                # --Model Section--
        model = tf.keras.models.load_model('./xception_trained/')

        # --Prediction--
        class_names = ['dark', 'green', 'light', 'medium']
        pred = model.predict(tf.expand_dims(img, axis  = 0))
        if len(pred[0]) > 1:
            pred_class = class_names[pred.argmax()] 
        else:
            pred_class = class_names[int(tf.round(pred)[0][0])]
        
        return render_template('index.html', prediction = pred_class, filename = filename)
    else:
        flash('Allowed image types are: jpeg, png and jpg')
        return redirect(request.url)
    
    

def allowed_file(filename):
    return '.'in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename = 'img/'+filename), code = 301)


if __name__ == '__main__':
    app.run(port = 3000, debug = True)


#gunicorn app:app -b :8080 --timeout 500 --workers=3 --threads=3 --worker-connections=1000