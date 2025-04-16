import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os

# Function to convert string to integer
def atoi(s):
    n = 0
    for i in s:
        n = n * 10 + ord(i) - ord("0")
    return n

# Creating directories for train and test sets for each emotion
outer_names = ['test', 'train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

base_dir = 'data'

# Creating directories if they don't exist
os.makedirs(base_dir, exist_ok=True)
for outer_name in outer_names:
    outer_path = os.path.join(base_dir, outer_name)
    os.makedirs(outer_path, exist_ok=True)
    for inner_name in inner_names:
        os.makedirs(os.path.join(outer_path, inner_name), exist_ok=True)

# Initialize emotion counters
emotion_count = {emotion: 0 for emotion in inner_names}
emotion_test_count = {emotion: 0 for emotion in inner_names}

# Load the FER-2013 dataset
df = pd.read_csv('./fer2013.csv')

# Image size (48x48)
mat = np.zeros((48, 48), dtype=np.uint8)

print("Saving images...")

# Iterate through each row in the CSV
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    
    # Fill the matrix with pixel values
    for j in range(2304):  # 48 * 48 = 2304 pixels
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    # Determine the target folder based on the emotion label
    emotion_label = df['emotion'][i]
    emotion_name = inner_names[emotion_label]

    # Train data: first 28709 samples
    if i < 28709:
        save_path = os.path.join(base_dir, 'train', emotion_name, f"im{emotion_count[emotion_name]}.png")
        emotion_count[emotion_name] += 1
    # Test data: remaining samples
    else:
        save_path = os.path.join(base_dir, 'test', emotion_name, f"im{emotion_test_count[emotion_name]}.png")
        emotion_test_count[emotion_name] += 1

    # Save the image
    img.save(save_path)

print("Done!")
