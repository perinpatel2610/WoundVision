import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from model import build_resnet152_unet
from tensorflow.keras.utils import Sequence
import albumentations as A

# ⚙ Augmentation
augment = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomBrightnessContrast(),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
])

# ⚙ Data Generator
class WoundDataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, file_list, batch_size=4, image_size=(512, 512), shuffle=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []

        for fname in batch_files:
            img = cv2.imread(os.path.join(self.image_dir, fname))
            mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, self.image_size)
            mask = cv2.resize(mask, self.image_size)

            # Augment
            augmented = augment(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

            img = img / 255.0
            mask = (mask > 127).astype(np.float32)

            images.append(img)
            masks.append(mask[..., np.newaxis])

        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_list)

# 📉 Dice Loss
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

# ➕ Combo Loss
def combined_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# 📢 Custom Logger
class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"📢 Epoch {epoch + 1} - "
              f"Loss: {logs['loss']:.4f} - Acc: {logs['accuracy']:.4f} - "
              f"Dice: {logs['dice_coef']:.4f} - IoU: {logs['iou_metric']:.4f} | "
              f"Val_Loss: {logs['val_loss']:.4f} - Val_Acc: {logs['val_accuracy']:.4f} - "
              f"Val_Dice: {logs['val_dice_coef']:.4f} - Val_IoU: {logs['val_iou_metric']:.4f}")


# Paths
img_dir = r"data_wound_seg\train_images" #Original Image
mask_dir = r"data_wound_seg\train_masks" #True Masked Image

# Split
all_filenames = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
train_files, val_files = train_test_split(all_filenames, test_size=0.2, random_state=42)

# Load generators
train_gen = WoundDataGenerator(img_dir, mask_dir, train_files, batch_size=4, image_size=(512, 512))
val_gen = WoundDataGenerator(img_dir, mask_dir, val_files, batch_size=4, image_size=(512, 512), shuffle=False)

# Build + Compile
model = build_resnet152_unet(input_shape=(512, 512, 3))
model.compile(optimizer=Adam(1e-4),
              loss=combined_loss,
              metrics=["accuracy", dice_coef, iou_metric])

# Callbacks
checkpoint = ModelCheckpoint("best_segmentation_model.h5", save_best_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
logger = EpochLogger()

# 🧠 Train
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[checkpoint, early_stop, logger],
    verbose=1
)
