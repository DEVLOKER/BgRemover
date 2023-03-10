
import json, uuid
import os, time
from flask import Flask, flash, request, redirect, abort, url_for, render_template, send_file
from waitress import serve
from BackgroundRemover import BackgroundRemover


PORT = 5000
app = Flask(__name__, static_folder='static', static_url_path='/')

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), BackgroundRemover.INPUT_PATH)
PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), BackgroundRemover.OUTPUT_PATH)
TMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), BackgroundRemover.TMP_PATH)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['TMP_FOLDER'] = TMP_FOLDER
app.config['SECRET_KEY'] = 'Sick Rat'

now = time.time() # datetime.utcnow().strftime("%Y%m%dT%H%M%S")

backgroundRemover = BackgroundRemover()


@app.route('/')
def index():
    cleanFiles()
    uploaded_files = __get_files(UPLOAD_FOLDER)
    processed_files = __get_files(PROCESSED_FOLDER)
    return render_template('index.html', uploaded_files=uploaded_files, processed_files=processed_files)




@app.route('/get/<code>', methods=['GET'])
def get(code):
    files = __get_files(UPLOAD_FOLDER)
    if code in files:
        path = os.path.join(UPLOAD_FOLDER, code)
        if os.path.exists(path):
            return send_file(path)
    abort(404)


@app.route('/download/<code>', methods=['GET'])
def download(code):
    files = __get_files(PROCESSED_FOLDER)
    if code in files:
        path = os.path.join(PROCESSED_FOLDER, code)
        if os.path.exists(path):
            return send_file(path)
    abort(404)



@app.route("/upload" , methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    #file = request.files['file']
    app.logger.info(request.files)
    upload_files = request.files.getlist('file')
    app.logger.info(upload_files)
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if not upload_files:
        flash('No selected file')
        return redirect(request.url)
    for file in upload_files:
        original_filename = file.filename
        extension = original_filename.rsplit('.', 1)[1].lower()
        filename = str(uuid.uuid1()) + '.' + extension
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    backgroundRemover.processAll()

    flash('Upload succeeded')
    return redirect(url_for('index'))



@app.route('/delete/<code>', methods=['GET'])
def delete(code):
    files = __get_files(UPLOAD_FOLDER)
    if code in files:
        path1 = os.path.join(UPLOAD_FOLDER, code)
        name = code.split('.')[0]
        path2 = os.path.join(PROCESSED_FOLDER, name + '.jpg')
        path3 = os.path.join(PROCESSED_FOLDER, name + '.png')
        path4 = os.path.join(TMP_FOLDER, name + '.jpg')
        path5 = os.path.join(PROCESSED_FOLDER, name + '_' + BackgroundRemover.RESULT_TYPE + '.webm')
        
        if os.path.exists(path1):
            os.remove(path1)
            os.path.exists(path2) and os.remove(path2)
            os.path.exists(path3) and os.remove(path3)
            os.path.exists(path4) and os.remove(path4)
            os.path.exists(path5) and os.remove(path5)
            return redirect(url_for('index'))
    abort(404)



def __get_files(path):
    files = {}
    # os.chdir(path)
    # listdir = sorted(filter(os.path.isfile, os.listdir(path)), key=os.path.getmtime)
    # print(listdir)
    name_list = os.listdir(path)
    full_list = [os.path.join(path,i) for i in name_list]
    time_sorted_list = sorted(full_list, key=os.path.getmtime)
    sorted_filename_list = [ os.path.basename(i) for i in time_sorted_list]
    sorted_filename_list.reverse()

    for filename in sorted_filename_list:
        files[filename] = filename
    return files
    # for filename in os.listdir(path):
    #     if os.path.isfile(os.path.join(path, filename)):
    #         files[filename] = filename
    # return files



def cleanFiles(): 
    timer = 60 * 60 # 60 minutes : 60 * 60      30 days : 30 * 86400
    for filename in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.getmtime(path) < now - timer:
            if os.path.isfile(path):
                os.remove(path)

    for filename in os.listdir(PROCESSED_FOLDER):
        path = os.path.join(PROCESSED_FOLDER, filename)
        if os.path.getmtime(path) < now - timer:
            if os.path.isfile(path):
                os.remove(path)




if __name__ == '__main__':
    # app.run('0.0.0.0', 5000, debug=True)
    serve(app, host="0.0.0.0", port=PORT)