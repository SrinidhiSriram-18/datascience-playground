import pandas as pd
import fasttext
import os

# Setting up the path
project_dir = os.path.dirname(os.path.dirname(__file__))
processed_data_dir = os.path.join(project_dir, 'data', 'processed')
models_dir = os.path.join(project_dir, 'models')
# Variable initialisation
space = " "

#load model
model_ft=fasttext.load_model(f'{models_dir}/model_ft.bin')

# Make predictions on new data
test_input = pd.read_csv("test_file.csv", sep=";")
test_input['review'] = test_input['review'].apply(lambda x: ' '.join(x).replace("\n", space))
test_input['rating'] = test_input['review'].apply(lambda x:model_ft.predict(x))
test_input['rating'] = test_input['rating'].apply(
    lambda x: int(str(x).replace("(('__label__","")[0]))


test_input[['reviewID', 'rating']].to_csv(f'{processed_data_dir}/predictions.csv', index=False)