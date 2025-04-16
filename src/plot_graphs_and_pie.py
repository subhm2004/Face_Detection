import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load training log
df = pd.read_csv("training_log.txt")

# Plot Accuracy & Loss Graphs
def plot_accuracy_loss(df):
    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(df['accuracy'], label='Train Accuracy')
    plt.plot(df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(df['loss'], label='Train Loss')
    plt.plot(df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("accuracy_loss_graph.png")
    plt.show()

# Dummy data for pie chart (replace with real emotion prediction counts if needed)
emotion_labels = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
counts = [300, 100, 150, 500, 400, 350, 200]  # replace these with actual predicted counts if available

def plot_emotion_pie(emotion_labels, counts):
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=emotion_labels, autopct='%1.1f%%', startangle=140)
    plt.title('Emotion Distribution')
    plt.savefig("emotion_pie_chart.png")
    plt.show()

# Use training log to simulate Decision Tree classifier (only demo purpose)
def train_decision_tree_from_log(df):
    features = df[['loss', 'val_loss', 'accuracy', 'val_accuracy']]
    labels = df['val_accuracy'] > 0.6  # dummy binary classification: good or bad accuracy

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    print("Classification Report (on training_log):")
    print(classification_report(y_test, clf.predict(X_test)))

    plt.figure(figsize=(10, 6))
    plot_tree(clf, feature_names=features.columns, class_names=['Low', 'High'], filled=True)
    plt.title("Decision Tree Classifier (based on training logs)")
    plt.savefig("decision_tree_visualization.png")
    plt.show()

# Call all plotting functions
plot_accuracy_loss(df)
plot_emotion_pie(emotion_labels, counts)
train_decision_tree_from_log(df)
