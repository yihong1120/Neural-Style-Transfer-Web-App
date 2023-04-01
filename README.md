# Neural-Style-Transfer-Web-App
This web app allows users to apply artistic styles to their own images or a set of images retrieved from Google Images using neural style transfer.

The app is built using Flask and Tensorflow, and utilizes the VGG19 convolutional neural network to extract feature representations of both the content and style images. The content and style losses are then computed and combined to create a new image that captures the content of the original image while applying the style of the style image.

## Getting Started
To use this app, clone the repository to your local machine:

    git clone https://github.com/yihong1120/neural-style-transfer.git

Navigate to the neural-style-transfer directory and create a new virtual environment:

    cd neural-style-transfer
    python3 -m venv venv

Activate the virtual environment:

    source venv/bin/activate

Install the required packages:

    pip install -r requirements.txt

To start the app, run:

    python app.py

Then, open your web browser and go to http://localhost:5000 to access the app.
