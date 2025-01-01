## Description
This application is designed to train a convolutional neural network to identify whether an image is a dog or cat in distributed mode. It contains a master server and 2 child servers. An interactive web app is designed in Streamlit to upload images and for visualisation. The images need to be uploaded in JPEG or PNG formats. The name of image file should contain either cat or dog in it for proper training The servers are implemented using Flask framework. The master server on initialisation sends a global copy of a CNN model to the child servers. When an image is uploaded to either one of the servers the child server trains the model on this image. It updates the weights. The updated weights are sent to the master server. The weights are then aggregated to get a new global model. This distributed training allows for privacy of images. The images are not shared to a common server and are stored locally. Only the weights are sent to the master server for processing.

## Installation
This project requires Python and a host of Python libraries installed
If you have Python installed but not the libraries, you can easily install them using 
pip install -r requirements.txt


## Usage
To use the application please execute the following commands in the order given below

•	python masterserver.py
•	python childserver1.py
•	python childserver2.py
•	streamlit run ./webapp.py

## Code Contributors


- Sreerag Chandran
