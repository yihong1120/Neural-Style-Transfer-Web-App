# Neural-Style-Transfer-Web-App
This web app allows users to apply artistic styles to their own images or a set of images retrieved from Google Images using neural style transfer.

The app is built using Flask and Tensorflow, and utilizes the VGG19 convolutional neural network to extract feature representations of both the content and style images. The content and style losses are then computed and combined to create a new image that captures the content of the original image while applying the style of the style image.

## Features

- Search images using Google, Bing, or Yahoo
- Upload a file for personalized search
- Assign a unique ID to keep track of search history
- Lookup previous search results using the unique ID
- Reset search history and uploaded files

## Technologies Used

- Python
- Flask
- HTML/CSS
- Bootstrap


## Getting Started
To use this app, clone the repository to your local machine:

```bash
git clone https://github.com/yihong1120/neural-style-transfer.git
```

Navigate to the neural-style-transfer directory

```bash
cd neural-style-transfer
```

Install the required packages:

```bash
pip install -r requirements.txt
```

To start the app, run:

```bash
python app.py
```

Then, open your web browser and go to http://localhost:5000 to access the app.

## Usage

### Reset
1. Click `Reset`.
2. Confirm the reset.

### Style Transfer
![Image search and style select section](https://github.com/yihong1120/Neural-Style-Transfer-Web-App/blob/main/images/search_setting.png)
1. Enter a search query for the style image or upload your own.
2. (Optional) Adjust the number of images to retrieve from Google Images.
3. (Optional) Select a search engine to use.
5. (Optional) Upload your own content image or use the default.
6. Click `Search`.
7. View the results.

![Display pictures grabbed from the Internet and the style template image](https://github.com/yihong1120/Neural-Style-Transfer-Web-App/blob/main/images/grabbed_images.png)
![Display the transfer images](https://github.com/yihong1120/Neural-Style-Transfer-Web-App/blob/main/images/transferred_images.png)

### Image Lookup
![Display lookup section](https://github.com/yihong1120/Neural-Style-Transfer-Web-App/blob/main/images/lookup.png)
1. Click `Lookup` in index page.
2. Enter a unique ID.
3. Click `Lookup`.
4. View the queries associated with the unique ID.
5. Click on a query to view the results.

## Customization
To customise the app, modify the parameters in app.py and style_transfer.py.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/yihong1120/Neural-Style-Transfer-Web-App/blob/main/LICENSE) file for details.
