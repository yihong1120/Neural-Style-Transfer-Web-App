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

    git clone https://github.com/yihong1120/neural-style-transfer.git

Navigate to the neural-style-transfer directory

    cd neural-style-transfer

Install the required packages:

    pip install -r requirements.txt

To start the app, run:

    python app.py

Then, open your web browser and go to http://localhost:5000 to access the app.

## Usage

### Style Transfer
1. Enter a search query for the style image or upload your own.
2. (Optional) Adjust the number of images to retrieve from Google Images.
3. (Optional) Select a search engine to use.
4. Click "Submit".
5. (Optional) Upload your own content image or use the default.
6. Click "Transform".
7. View the results.

### Image Lookup
1. Click "Lookup".
2. Enter a unique ID.
3. Click "Submit".
4. View the queries associated with the unique ID.
5. Click on a query to view the results.

### Reset
1. Click "Reset".
2. Confirm the reset.

## Customization
To customise the app, modify the parameters in app.py and style_transfer.py.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/yihong1120/Neural-Style-Transfer-Web-App/blob/main/LICENSE) file for details.
