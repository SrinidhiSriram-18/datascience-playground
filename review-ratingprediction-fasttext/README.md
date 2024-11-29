## Objective
--------------
To design and implement a model that can effectively predict sentiment ratings (1-5) for social media comments based on Wongnai's existing review data.

## Requirements
---------------
* Python 3.8+
* FastText library
* Scikit-learn library
* Pandas library
* pythainlp library

## Installation
---------------
1. Install requirements:``` pip install -r requirements.txt```
2. Train the model: ```python src/predict.py```
3. Evaluate the model: ```python src/evaluate.py```

## Solutions

## Approach 1 (Attempted)
### Description
---------------
Attempted to translate thai language to english using libraries for better understanding and text normalising converting all the letters to lower case to avoid treating the same word differently based on caseto avoid treating the same word differently based on case and then use pretrained models on the translated data.

### Usage
---------
1. Prepare your dataset in CSV format
2. Run attempted_translation.py to train the model

### challenge faced
---------------------
1. googletrans library - limited translation ability - faced timeout after processing around 300 row.
2. Google Translate API,DeepL API,Microsoft Translator - required paid subscription

## Approach 2 (Implemented)
### Description
---------------
Built prediction model using fasttext library considering the following :
* Efficient Training of  large datasets.
* Fast Prediction capabilities.
* Support for Multiple Languages
* Handling Out-of-Vocabulary (OOV) Words
* Simple Architecture

### Features
--------------------
* FastText-based text prediction model
* Supports multiple text categories
* Achieves high F1 score accuracy

### Directions to use
---------------------------
1. Prepare your dataset in CSV format
2. Run src/predict.py to train the model, predict the outcome and write to processed folder. 
3. Run src/evaluate.py to evaluate the model. 

### Model Performance
---------------------
* Mean F1 Score: 0.415
* accuracy of 57.9% 

### Output 
---------------------
Once the code for evaluation is completed, output file gets created as predictions.csv in data/processed