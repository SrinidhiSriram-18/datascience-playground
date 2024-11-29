import fasttext
import pandas as pd
from stopwords import get_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Setting up the path
project_dir = os.path.dirname(os.path.dirname(__file__))
raw_data_dir = os.path.join(project_dir, 'data', 'raw')
processed_data_dir = os.path.join(project_dir, 'data', 'processed')
models_dir = os.path.join(project_dir, 'models')

# Load data
df = pd.read_csv(f'{raw_data_dir}/w_review_train.csv', delimiter=';',names=['review','rating'])

# Get Thai stopwords
stopwords = get_stopwords("th")

# Function to remove stopwords from a text
def remove_stopwords(text):
    tokens = word_tokenize(text)  # Tokenize the text into words
    filtered_tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(filtered_tokens)

# Preprocess text
# df['review'] = df['review'].apply(word_tokenize)
# Apply the remove_stopwords function to the 'Comment' column
df['review'] = df['review'].apply(remove_stopwords)
df['review'] = df['review'].apply(lambda x: ' '.join(x))

# Split data
X_train, X_val, y_train, y_val = train_test_split(df['review'], df['rating'], test_size=0.2, random_state=42)


# Save training data to file
space = " "
with open(f'{processed_data_dir}/train.ft', 'w') as f:
    for review, rating in zip(X_train, y_train):
        f.write('__label__{} {}\n'.format(rating, review.replace("\n", space)))
# Save Test data 
with open(f'{processed_data_dir}/test.ft', 'w') as f:
    for review, rating in zip(X_val, y_val):
        f.write('__label__{} {}\n'.format(rating, review.replace("\n", space)))



# Train FastText Model
model_ft = fasttext.train_supervised(f'{processed_data_dir}/train.ft', epoch=25,
    lr=1,
    wordNgrams=5,
    dim=500,
    thread=16,
    bucket=1000000,
)
    #loss='so')
model_ft.save_model(f'{models_dir}/model_ft.bin')


# Make predictions on validation set
y_pred_ft_ratings = []
for review in X_val.tolist():
    review = str(review)
    review = review.replace("\n", space)
    labels, probs = model_ft.predict(review)
    rating = int(labels[0].replace('__label__', ''))
    y_pred_ft_ratings.append(rating)


# Evaluate Model
mse_ft = mean_squared_error(y_val, y_pred_ft_ratings)
print('FastText MSE: {:.4f}'.format(mse_ft))

# Evaluate the model on test data
# Save test data to file
accuracy = model_ft.test(f'{processed_data_dir}/test.ft',k=1)
print("Accuracy:", accuracy)
# # Calculate evaluation metrics
accuracy = accuracy_score(y_val, y_pred_ft_ratings)
print('Accuracy:', accuracy)
report = classification_report(y_val, y_pred_ft_ratings, output_dict=True)

# # Print the classification report for more detailed metrics
# print(report)

# # Calculate the mean F1 score (macro average)
mean_f1 = report['macro avg']['f1-score']
print(f"Mean F1 Score: {mean_f1:.4f}")


