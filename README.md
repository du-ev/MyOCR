# MyOCR
Deep learning OCR implementation with CRNN (CNN-RNN-CTC) for identifying written and synthetic text, experimentally summarizing identified text using Google Gemini API.

## Getting Started:
Create a new conda environment: ```conda create --name FinalCapEnv```

Activate your new conda environment: ```conda activate FinalCapEnv```

Run in terminal/environment: ```pip install streamlit nltk opencv-python Pillow numpy torch torchvision tqdm python-dotenv dill kagglehub google-generativeai``` 
* ```streamlit``` for UI
* ```nltk``` for tokenization
* ```opencv-python``` for image processing
* ```Pillow``` for simplified image handling
* ```numpy``` for vectorization and data processing
* ```torch``` and ```torchvision``` for OCR
* ```tqdm``` for progress bars
* ```python-dotenv``` for API processing (IP)
* ```dill``` for file saving in .pkl
* ```kagglehub``` for accessing Kaggle datasets
* ```google-genai``` for summarization of CRNN text output

Pregenerate data by running ```main.py```.
A Gemini AI API key will need to be created and stored in ```.env```.

## Training Model:

Train by running ```train.py```. Adjust default hyperparameters in ```hyperparameters.py```.