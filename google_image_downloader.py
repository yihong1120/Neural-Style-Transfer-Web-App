import os
import re
import requests
from google_images_search import GoogleImagesSearch


class ImageDownloader:
    def __init__(self, api_key, cx_id):
        self.api_key = api_key
        self.cx_id = cx_id
        self.gis = GoogleImagesSearch(api_key, cx_id)

    def search_images(self, query, num_images=5, img_size='large', file_type='jpg|png|webp|jpeg|gif'):
        search_params = {
            'q': query,
            'num': num_images,
            'imgSize': img_size,
            'fileType': file_type,
        }
        self.gis.search(search_params)
        return self.gis.results()

    def download_image(self, image, index, path):
        response = requests.get(image.url, stream=True)
        if response.status_code == 200:
            file_extension = re.findall(r'\.([a-zA-Z0-9]+)(\?|$)', image.url)
            if file_extension:
                file_extension = file_extension[0][0]
                with open(os.path.join(path, f"image_{index + 1}.{file_extension}"), 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)

    def download_images(self, query, num_images=5, unique_id=None):
        results = self.search_images(query, 20)  # 增加搜索結果數量，以確保可以找到至少 5 張非 SVG 圖像

        # Include the unique_id in the image_folder path
        image_folder = os.path.join('static', unique_id, query) if unique_id else os.path.join('static', query)
        os.makedirs(image_folder, exist_ok=True)

        index = 0
        for image in results:
            file_extension = re.findall(r'\.([a-zA-Z0-9]+)(\?|$)', image.url)
            if file_extension:
                file_extension = file_extension[0][0].lower()
                if file_extension != 'svg':
                    self.download_image(image, index, image_folder)
                    index += 1
                    if index >= num_images:
                        break
        return image_folder


if __name__ == "__main__":
    api_key = 'AIzaSyD1aoB_kiPTfMpRESqpigmUaSjq1Puz9Gk'
    cx_id = '62842d2c8c40d4f4c'
    query = 'puppies'
    num_images = 5

    downloader = ImageDownloader(api_key, cx_id)
    image_folder = downloader.download_images(query, num_images, unique_id)
    print(f"Downloaded images to {image_folder}")
