import os
import re
import requests
from google_images_search import GoogleImagesSearch
from typing import List


class ImageDownloader:
    def __init__(self, api_key: str, cx_id: str):
        """
        Initialises the ImageDownloader class.

        Args:
            api_key (str): The Google API key.
            cx_id (str): The custom search engine ID.
        """
        self.api_key = api_key
        self.cx_id = cx_id
        self.gis = GoogleImagesSearch(api_key, cx_id)

    def search_images(self, query: str, num_images: int=5, img_size: str='large', file_type: str='jpg|png|webp|jpeg|gif') -> List:
        """
        Searches for images based on the query.

        Args:
            query (str): The search query.
            num_images (int, optional): The number of images to search for. Defaults to 5.
            img_size (str, optional): The image size to search for. Defaults to 'large'.
            file_type (str, optional): The file types to search for. Defaults to 'jpg|png|webp|jpeg|gif'.

        Returns:
            List: The search results.
        """
        search_params = {
            'q': query,
            'num': num_images,
            'imgSize': img_size,
            'fileType': file_type,
        }
        self.gis.search(search_params)
        return self.gis.results()

    def download_image(self, image, index: int, path: str):
        """
        Downloads an image.

        Args:
            image: The image to download.
            index (int): The index of the image.
            path (str): The path to save the image.
        """
        response = requests.get(image.url, stream=True)
        if response.status_code == 200:
            file_extension = re.findall(r'\.([a-zA-Z0-9]+)(\?|$)', image.url)
            if file_extension:
                file_extension = file_extension[0][0]
                with open(os.path.join(path, f"image_{index + 1}.{file_extension}"), 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)

    def download_images(self, query: str, num_images: int=5, unique_id: str=None) -> str:
        """
        Downloads images based on the query.

        Args:
            query (str): The search query.
            num_images (int, optional): The number of images to download. Defaults to 5.
            unique_id (str, optional): The unique ID for the download. Defaults to None.

        Returns:
            str: The path where the images are saved.
        """
        results = self.search_images(query, 20)  # Increase the number of search results to ensure at least 5 non-SVG images can be found

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
    api_key = ''
    cx_id = ''
    query = 'puppies'
    num_images = 5

    downloader = ImageDownloader(api_key, cx_id)
    image_folder = downloader.download_images(query, num_images, unique_id)
    print(f"Downloaded images to {image_folder}")
