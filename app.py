from flask import Flask, render_template, request, redirect, url_for
from google_images_search import GoogleImagesSearch
from style_transfer import StyleTransfer
from google_image_downloader import ImageDownloader
from PIL import Image
import os, re, requests, shutil, uuid
import sched, time
from datetime import datetime, timedelta

app = Flask(__name__)

class ImageHandler:
    def __init__(self, api_key, cx_id):
        self.downloader = ImageDownloader(api_key, cx_id)

    def download_images(self, query, num_images, unique_id):
        image_folder = self.downloader.download_images(query, num_images, unique_id)
        return image_folder

    def transform_images(self, unique_id, query):
        max_dim, content_weight, style_weight = 512, 1e-5, 1e3
        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        input_folder = os.path.join("static", unique_id, query)
        style_path = os.path.join('static', unique_id, query, 'style_index.jpg')
        style_transfer = StyleTransfer(None, style_path, max_dim, content_weight, style_weight, content_layers, style_layers)
        style_transfer.process_folder(input_folder, style_path)

class FolderHandler:
    @staticmethod
    def clear_folder(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def cleanup_folders(self):
        static_folder = 'static'
        unique_id_folders = os.listdir(static_folder)
        for unique_id in unique_id_folders:
            unique_id_path = os.path.join(static_folder, unique_id)
            if os.path.isdir(unique_id_path):
                query_folders = os.listdir(unique_id_path)
                for query in query_folders:
                    query_path = os.path.join(unique_id_path, query)
                    if os.path.isdir(query_path):
                        folder_creation_time = datetime.fromtimestamp(os.path.getctime(query_path))
                        folder_age = datetime.now() - folder_creation_time
                        if folder_age > timedelta(days=30):
                            shutil.rmtree(query_path)
                            print(f"Deleted {query_path} (age: {folder_age})")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query, num_images, search_engine, unique_id = handle_form(request)
        if not query:
            return render_template('index.html', error='Please enter a search query.')

        api_key = 'AIzaSyD1aoB_kiPTfMpRESqpigmUaSjq1Puz9Gk'
        cx_id = '62842d2c8c40d4f4c'
        image_handler = ImageHandler(api_key, cx_id)
        image_folder = image_handler.download_images(query, num_images, unique_id)

        file = request.files.get('file')
        handle_file_upload(file, unique_id, query)

        image_handler.transform_images(unique_id, query)

        images =sorted(os.listdir(image_folder))
        return render_template('index.html', images=images, query=query, unique_id=unique_id)
    
    return render_template('index.html')

@app.route('/reset', methods=['GET'])
def reset():
    FolderHandler.clear_folder('static')
    return redirect(url_for('index'))

@app.route('/lookup', methods=['GET', 'POST'])
def lookup():
    if request.method == 'POST':
        unique_id = request.form.get('unique_id')
        if not unique_id:
            return render_template('lookup.html', error='Please enter a unique_id.')
        folder = os.path.join('static', unique_id)
        if os.path.exists(folder):
            queries = os.listdir(folder)
            return render_template('lookup.html', queries=queries, unique_id=unique_id)
        else:
            return render_template('lookup.html', error='Invalid unique_id.')

    return render_template('lookup.html')

@app.route('/lookup/<unique_id>/<query>', methods=['GET'])
def lookup_results(unique_id, query):
    folder = os.path.join('static', unique_id, query)
    if os.path.exists(folder):
        images = sorted(os.listdir(folder))
        return render_template('index.html', images=images, query=query, unique_id=unique_id)
    else:
        return render_template('lookup.html', error='No results found for the given unique_id and query.')

def handle_form(request):
    query = request.form.get('query')
    num_images = int(request.form.get('num_images', 5))
    search_engine = request.form.get('search_engine')
    unique_id = request.form.get('unique_id') or str(uuid.uuid4())
    return query, num_images, search_engine, unique_id

def handle_file_upload(file, unique_id, query):
    if file:
        filename = file.filename
        file.save(os.path.join('static', unique_id, query, 'style_index.jpg'))
    else:
        shutil.copyfile("Vogh.jpg", os.path.join('static', unique_id, query, 'style_index.jpg'))

if __name__ == '__main__':
    scheduler = sched.scheduler(time.time, time.sleep)
    folder_handler = FolderHandler()

    def schedule_cleanup(sc):
        folder_handler.cleanup_folders()
        scheduler.enter(86400, 1, schedule_cleanup, (sc,))

    scheduler.enter(86400, 1, schedule_cleanup, (scheduler,))
    scheduler.run(blocking=False)

    app.run(debug=True)