# keras-arbitrary-style-transfer
An attempt to reproduce the results of Ghiasi et al 2017 "Exploring the structure of a real-time, arbitrary neural artistic stylization network" with Keras layers

# Usage
To train the model, put content images in content_images and style images to style_images
`python3 train.py model_name`

To run the model (i.e. re-style image):
`python3 run.py model_name content_img style_img`

To run the original, optimization based style transfer based on Gatys2015 that requires no model
`python3 single.py content_img style_img`

# Disclaimer

# What I've learned doing this
* Tricks needed to get models working with no labels and no input (single)
* Colors - make sure you limit to range 0-255
* Weighting between style and content is important - read original Gatys2015 for good illustrations
* Instance normalization is really handwaved in Ghiasi2017. Read Dumoulin2016 instead for that.
* Normalization/scaling of loss functions also matters (esp since different layers have different magnitude activations)
* Decent models take *a lot* of time to train. As in days.
* Optimization based system is good for testing the loss functions
* Total variation loss is really important in opt based system to get it to converge to a smooth image
* Activation (relu) goes *after* normalization 

* Getting tensorflow graph saved is annoying. https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/ helped
* Tensorflow 1 vs 2 is different, and 2 requires you use tf.keras everywhere or crashes with non-descriptive error message "NameError: free variable 'do_return_2' referenced before assignment in enclosing scope"
