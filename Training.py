TRAIN_DATASET_PATH = 'data\train\BraTS2020_TrainingData'

import os
import numpy as np
from data_generator import imageLoader
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random
from tensorflow.keras import backend as K
import segmentation_models_3D as sm
from keras.metrics import MeanIoU
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.models import load_model
from PIL import Image


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


tf.keras.backend.set_floatx('float32')


train_img_dir = TRAIN_DATASET_PATH + "/train/images/"
train_mask_dir = TRAIN_DATASET_PATH + "/train/masks/"


val_img_dir = TRAIN_DATASET_PATH + "/val/images/"
val_mask_dir = TRAIN_DATASET_PATH + "/val/masks/"


train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)


val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

batch_size = 5


train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list, 
                              val_mask_dir, val_mask_list, batch_size)


img, msk = train_img_datagen.__next__()


img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask, axis=3)


n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))


plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

wt0, wt1, wt2, wt3 = np.float32(0.25), np.float32(0.25), np.float32(0.25), np.float32(0.25)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3], dtype=np.float32))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


def calculate_all_metrics(y_true, y_pred)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')
    precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    sensitivity = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    

    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    specificity = tn / (tn + fp + 1e-7)
    
    
    intersection = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    union = np.sum((y_true_flat == 1) | (y_pred_flat == 1))
    iou = intersection / (union + 1e-7)
    
    
    dice = 2 * intersection / (np.sum(y_true_flat) + np.sum(y_pred_flat) + 1e-7)
    
    return {
        'f1': f1,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'iou': iou,
        'dice': dice
    }


metrics = [
    'accuracy',
    sm.metrics.IOUScore(threshold=0.5),
    sm.metrics.FScore(threshold=0.5),
    precision_metric,
    recall_metric,
    f1_score_metric,
    iou_metric,
    dice_metric
]


def plot_metrics(history):
    """Eğitim ve doğrulama metriklerini çiz"""
    for metric_name in history.keys():
        plt.figure(figsize=(10, 6))
        plt.plot(history[metric_name], 'y', label=f'Train {metric_name}')
        if f'val_{metric_name}' in history:
            plt.plot(history[f'val_{metric_name}'], 'r', label=f'Validation {metric_name}')
        plt.title(f'Training and Validation {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.show()

def plot_precision_recall_f1(history):
    """Precision, Recall ve F1 Score grafiğini çiz"""
    if 'precision_metric' in history and 'recall_metric' in history and 'f1_score_metric' in history:
        plt.figure(figsize=(10, 6))
        
      
        plt.plot(history['recall_metric'], history['precision_metric'], 'b', label='Precision vs Recall')
        
       
        plt.plot(history['recall_metric'], history['f1_score_metric'], 'g', label='F1 Score vs Recall')
        
        plt.title('Precision and F1 Score vs Recall')
        plt.xlabel('Recall')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

LR = 0.0001
optim = keras.optimizers.Adam(LR)


steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

from Unet_model import simple_unet_model


model = simple_unet_model(IMG_HEIGHT=128, 
                          IMG_WIDTH=128, 
                          IMG_DEPTH=128, 
                          IMG_CHANNELS=3, 
                          num_classes=4)


model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())


epochs = 1
history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch)

model_filename = f'saved_models/brats_3d_{epochs}epochs_simple_unet_weighted_dice.hdf5'
model.save(model_filename)


plot_metrics(history.history)


plot_precision_recall_f1(history.history)


my_model = load_model(model_filename, 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score': sm.metrics.IOUScore(threshold=0.5), 
                                      'f_score': sm.metrics.FScore(threshold=0.5),
                                      'iou_metric': iou_metric,
                                      'dice_metric': dice_metric,
                                      'f1_score_metric': f1_score_metric,
                                      'precision_metric': precision_metric,
                                      'recall_metric': recall_metric})
my_model.compile(optimizer=keras.optimizers.Adam(LR), loss=total_loss, metrics=metrics)


test_img_datagen = imageLoader(val_img_dir, val_img_list, 
                               val_mask_dir, val_mask_list, batch_size)


test_image_batch, test_mask_batch = test_img_datagen.__next__()


test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)


n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


img_num = 42
test_img = np.load(TRAIN_DATASET_PATH + "/val/images/image_" + str(img_num) + ".npy")
test_mask = np.load(TRAIN_DATASET_PATH + "/val/masks/mask_" + str(img_num) + ".npy")
test_mask_argmax = np.argmax(test_mask, axis=3)


test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]


for metric in metrics:
    plot_metrics(history.history)

n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_argmax[:, :, n_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:, :, n_slice])
plt.show()


def preprocess_image(image_path, target_size):
    if image_path.endswith('.npy'):
        img = np.load(image_path)
    else:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img = np.array(img)
    
    img = img.astype(np.float32) / 255.0 
    img = np.expand_dims(img, axis=0)  
    return img


def predict_and_visualize(model, image_path, mask_path=None, target_size=(128, 128, 128)):

    img = preprocess_image(image_path, target_size)
    
    
    prediction = model.predict(img)
    prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]
    

    n_slice = prediction_argmax.shape[2] // 2  
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(img[0, :, :, n_slice, 0], cmap='gray')
    
    plt.subplot(122)
    plt.title('Predicted Mask')
    plt.imshow(prediction_argmax[:, :, n_slice])
    
    plt.show()
    
   
    if mask_path:
        mask = preprocess_image(mask_path, target_size)
        mask_argmax = np.argmax(mask, axis=4)[0, :, :, :]
        metrics = calculate_all_metrics(mask_argmax, prediction_argmax)
        for metric_name, value in metrics.items():
            print(f'{metric_name}: {value:.4f}')
        
        return metrics

# Example usage
#image_path = 'path/to/your/image.npy'  # Update with your image path
#mask_path = 'path/to/your/mask.npy'  # Update with your mask path if available
#metrics = predict_and_visualize(my_model, image_path, mask_path)

plt.figure(figsize=(10, 6))
plt.plot(history.history['iou_score'], 'y', label='Train Mean IoU')
plt.plot(history.history['val_iou_score'], 'r', label='Validation Mean IoU')
plt.title('Training and Validation Mean IoU')
plt.xlabel('Epoch')
plt.ylabel('Mean IoU')
plt.legend()
plt.show()
