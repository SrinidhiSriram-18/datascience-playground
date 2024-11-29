import pandas as pd
import time

# using google trans -> encountered timeout exception 
from googletrans import Translator
translator = Translator(timeout=600)
def thai_to_english(text):
    translator = Translator()
    translation = translator.translate(text, dest='en', src='th')
    return translation.text

# Importing data
Train_Data = pd.read_csv('w_review_train.csv', delimiter=';')

# Getting data stats
next_column_name = len(Train_Data.columns)
text=Train_Data.columns[0]
total_rows = len(Train_Data)

# Initialising variables
english_text={}
chunk_size=100 
wait_time=5

print("total rows:",total_rows,'  chunk_size:',chunk_size,'    total_rows:',total_rows)
# for i, line in enumerate(Train_Data[text]):
    # print (line)
    # print(i)
# splitting the file into small files due toprocessing 

    
# process in chunks of 50 lines due to time exceeded issue    
for start_row in range(0, total_rows, chunk_size):  
    end_row = min(start_row + chunk_size, total_rows)
    chunk = Train_Data[text].iloc[start_row:end_row]
    print(f"Processing rows {start_row+1} to {end_row}...")
    for i, line in enumerate(chunk):
       english_text[i] = thai_to_english(line)
    # Wait for 5 seconds before processing the next chunk
    if end_row < total_rows:
        print(f"Waiting for {wait_time} seconds before processing next chunk...")
        time.sleep(wait_time)

print(english_text[i])
Train_Data[next_column_name]=english_text
Train_Data.to_csv('Translated_data.csv', sep=',', index=False,header=False)
print("finish")



thai_text = "ร้านเค้กน่ารักๆ ตรงชั้นล่างของห้างเซ็นทรัลลาดพ"
# english_text = thai_to_english(thai_text)
# print(english_text)  # Output: Hello