# Image Captioning Using Neural Networks

The image captioning code used in this project was adopted from Hvass Lab at https://github.com/Hvass-Labs/TensorFlow-Tutorials from his notebook 22_Image_Captioning.ipynb .  

Since the model runs best on GPU, we recommend running it on Google Colab.

## Running the code

To run the original model along with testing it on our LMU dataset: \
-Download the image_captioning_lmudata.ipynb file \
-Open in Google Drive \
-Change runtime type to GPU \
-Run all cells

To run the model with an LSTM instead of an RNN, download the Image_Captioning_LSTM.ipynb and follow the same steps.

## Alternative model

The rest of the image captioning code (Persson model) was adapted from the Persson code found here: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning  
Additionally, more notes on running the files and further details on Andrew's part of the project can be found at this repo link: https://github.com/ABruneel04/NLP_Final_Project  

## Additional files

install_coco.sh - bash script to help install the COCO dataset \
project.py - code for downloading the Cifar-10 dataset
