import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# Load dataset
df = pd.read_csv("fer2013.csv")

# Create directory structure
emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
base_path = 'emotion_data'

for usage in ['Training', 'PublicTest']:
    for emotion in emotions:
        os.makedirs(os.path.join(base_path, 'train' if usage == 'Training' else 'test', emotion), exist_ok=True)

# Convert pixels to image and save
for i in tqdm(range(len(df))):
    row = df.iloc[i]
    pixels = np.fromstring(row['pixels'], dtype=int, sep=' ').reshape(48, 48)
    emotion = emotions[int(row['emotion'])]
    usage = 'train' if row['Usage'] == 'Training' else 'test'

    path = os.path.join(base_path, usage, emotion, f"{i}.jpg")
    cv2.imwrite(path, pixels)
