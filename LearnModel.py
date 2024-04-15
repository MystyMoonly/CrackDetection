import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import plotly.express as px

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, r2_score

import tensorflow as tf

from keras.applications import VGG16
from keras.layers import Dropout


### Get Positive & Negative Directories 

positive_dir = Path(r'C:\Users\Mysty\Downloads\DATA_Maguire_20180517_ALL\P\CP') #Cracked
negative_dir = Path(r'C:\Users\Mysty\Downloads\DATA_Maguire_20180517_ALL\P\UP') #Uncracked

### Creating DataFrames

def generate_df(img_dir, label):
    
    file_paths = pd.Series(list(img_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=file_paths.index)
    df = pd.concat([file_paths, labels], axis=1)
    
    return df

positive_df = generate_df(positive_dir, 'POSITIVE') 
# 'POSITIVE' обозначает наличие трещины на изображении.
negative_df = generate_df(negative_dir, 'NEGATIVE')
# 'NEGATIVE' обозначает отсутствие трещины на изображении.

# concatenate both positive and negative df
all_df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1, random_state=1).reset_index(drop=True)
all_df

### Split the DataSet

train_df, test_df = train_test_split(all_df.sample(5200, random_state=1), 
                train_size=0.6,
                shuffle=True,
                random_state=1)


# Add data augmentation
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                            validation_split=0.2,
                                                            rotation_range=20,
                                                            width_shift_range=0.2,
                                                            height_shift_range=0.2,
                                                            shear_range=0.2,
                                                            zoom_range=0.2,
                                                            horizontal_flip=True,
                                                            fill_mode='nearest')

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_dataframe(train_df, 
                                          x_col='Filepath',
                                          y_col='Label',
                                          target_size=(120,120), 
                                          color_mode='rgb',
                                          class_mode='binary',
                                          batch_size=32,
                                          shuffle=True,
                                          seed=42,
                                          subset='training')


val_data = train_gen.flow_from_dataframe(train_df, 
                                          x_col='Filepath',
                                          y_col='Label',
                                          target_size=(120,120), 
                                          color_mode='rgb',
                                          class_mode='binary',
                                          batch_size=32,
                                          shuffle=True,
                                          seed=42,
                                          subset='validation')


test_data = test_gen.flow_from_dataframe(test_df, 
                                          x_col='Filepath',
                                          y_col='Label',
                                          target_size=(120,120), 
                                          color_mode='rgb',
                                          class_mode='binary',
                                          batch_size=32,
                                          shuffle=False,
                                          seed=42)

test_data

# ### Training DataSet

# Use pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(120, 120, 3))
base_model.trainable = False

inputs = tf.keras.Input(shape=(120,120,3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# print model summary
model.summary()

# Training the model
history = model.fit(train_data, validation_data=val_data, epochs=100, 
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=3,
                                                          restore_best_weights=True)
                         ])


model_path = 'trained_model_new.h5'

# Check if the file exists
if os.path.isfile(model_path):
    # If it exists, remove it
    os.remove(model_path)

# Save the model
model.save('trained_model_new.h5', overwrite=True)

### Plotting

fig = px.line(history.history,
             y=['loss', 'val_loss'],
             labels={'index':'Epoch'},
             title='Training and Validation Loss over Time')

fig.show()


def evaluate_model(model, test_data):
    
    results = model.evaluate(test_data, verbose=0)
    loss = results[0]
    accuracy = results[1]
    
    print(f'Test Loss {loss:.5f}')
    print(f'Test Accuracy {accuracy * 100:.2f} %')
    
    
    # predicted y values
    y_pred = np.squeeze((model.predict(test_data) >= 0.3).astype(int))
    y_certain = np.squeeze((model.predict(test_data)).astype(int))
    
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

    ### Final Results
evaluate_model(model, test_data)






