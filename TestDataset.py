import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, r2_score
import tensorflow as tf
from keras.applications import VGG16
from keras.layers import Dropout
from keras.models import load_model

def generate_df(img_dir):
    file_paths = pd.Series(list(img_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    # Define the label for each image based on its filename
    labels = file_paths.apply(lambda x: 'POSITIVE' if 'cracked' in x else 'NEGATIVE')
    df = pd.concat([file_paths, labels.rename('Label')], axis=1)
    return df

# Define the directory where your test images are located
test_dir = Path(r'C:\Users\Mysty\Documents\GitHub\machineLearningVSCode_BeforeAmmoClear\agoorCrackDetection\new\check')

# Create a DataFrame for the test images
test_df = generate_df(test_dir)

# Load the model
model = load_model('trained_model_new.h5')

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_dataframe(test_df, 
                                          x_col='Filepath',
                                          y_col='Label',
                                          target_size=(120,120), 
                                          color_mode='rgb',
                                          class_mode='binary',
                                          batch_size=32,
                                          shuffle=False,
                                          seed=42)

def evaluate_model(loaded_model, test_data):
    results = loaded_model.evaluate(test_data, verbose=0)
    loss = results[0]
    accuracy = results[1]
    print(f'Test Loss {loss:.5f}')
    print(f'Test Accuracy {accuracy * 100:.2f} %')
    y_pred_certain = loaded_model.predict(test_data)
    y_pred = np.squeeze((y_pred_certain >= 0.75).astype(int))
    conf_matr = confusion_matrix(test_data.labels, y_pred)
    class_report = classification_report(test_data.labels, y_pred,
                                         target_names=['NEGATIVE', 'POSITIVE'])
    plt.figure(figsize=(6,6))
    sns.heatmap(conf_matr, fmt='g', annot=True, cbar=False, vmin=0, cmap='Blues')
    plt.xticks(ticks=np.arange(2) + 0.5, labels=['NEGATIVE', 'POSITIVE'])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=['NEGATIVE', 'POSITIVE'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print('r2 Score : ', r2_score(test_data.labels, y_pred))
    print()
    print('Classification Report :\n......................\n', class_report)

evaluate_model(model, test_data)

  ### Testing New DataSet

def test_new_data(dir_path):
    
    new_test_dir = Path(dir_path)
    
    df_new = generate_df(new_test_dir)  # Use one of the labels your model was trained on
    
    test_data_new = test_gen.flow_from_dataframe(df_new, 
                                          x_col='Filepath',
                                          target_size=(120,120), 
                                          color_mode='rgb',
                                          batch_size=5,
                                          shuffle=False,
                                          seed=42,
                                          class_mode=None)  # Set class_mode=None
    
    # Get filenames
    filenames = test_data_new.filenames

    # Get only the base name of the file
    filenames = [os.path.basename(f) for f in filenames]
    
    # predicted y values
    y_pred_certain = model.predict(test_data_new).round(6) # Call model.predict once
    y_pred = y_pred_certain.flatten() # Flatten the array to make sure it's 1D

    y_out = []
    
    for i in y_pred:
        if i >= 0.75:
            y_out.append('Positive(Crack) ')
        else:
            y_out.append('Negative (Not Crack)')

    # Create DataFrame with filenames, results, and confidence
    result = pd.DataFrame(np.c_[filenames, y_out, y_pred_certain], columns=['Filename', 'Result', 'Confidance of being Cracked'])
    return result

results = test_new_data(r'C:\Users\Mysty\Documents\GitHub\machineLearningVSCode_BeforeAmmoClear\agoorCrackDetection\new\check')

results.to_csv('final_results.csv')