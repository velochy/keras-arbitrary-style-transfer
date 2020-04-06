# keras-arbitrary-style-transfer
An attempt to reproduce the results of Ghiasi et al 2017 "Exploring the structure of a real-time, arbitrary neural artistic stylization network" with Keras layers

# Usage
To train the model, put content images in content_images and style images to style_images
`python3 train.py model_name`

To run the model (i.e. re-style image):
`python3 run.py model_name content_img style_img`



# Disclaimer
Author is currently not convinced this works, as 8 hours of training on a GPU does not lead to the correct style being transfered yet.