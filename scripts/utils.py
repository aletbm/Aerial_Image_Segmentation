import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import cv2
from tensorflow.image import (resize_with_pad,
                              resize,)

def load_img(filename):
    image = tf.convert_to_tensor(cv2.imread(filename)[..., ::-1])
    return image

def get_data(df, dataset=None):
    if dataset is None:
        num_images = len(df)
        random_idx = random.randint(0, num_images - 1)
        filename_img, filename_mask, dt_abb = df.loc[random_idx]
    else:
        df_aux = df[df["dataset_abb"] == dataset]
        num_images = len(df_aux)
        random_idx = random.randint(0, num_images - 1)
        filename_img, filename_mask, dt_abb = df_aux.iloc[random_idx]

    image = load_img(filename_img)
    mask = load_img(filename_mask)
    return image, mask, dt_abb

def norm_colors(mask, colors, colors_conv):
    colors = tf.cast(colors, dtype=tf.uint8)
    colors_conv = tf.cast(colors_conv, dtype=tf.uint8)
    n_colors = tf.size(colors)//3
    boolean_all = tf.TensorArray(dtype=tf.bool, size=n_colors)

    for i in tf.range(n_colors):
        x, y = colors[i], colors_conv[i]
        boolean_mask = tf.reduce_all(tf.equal(mask, x), axis=-1)
        boolean_all = boolean_all.write(i, boolean_mask)
        r, g, b = y[0], y[1], y[2]
        R = tf.where(boolean_mask, r, mask[..., 0])
        G = tf.where(boolean_mask, g, mask[..., 1])
        B = tf.where(boolean_mask, b, mask[..., 2])
        mask = tf.stack([R, G, B], axis=-1)

    boolean_all = tf.reduce_any(boolean_all.stack(), axis=0)
    unlabeled = tf.constant((0, 0, 0), dtype=tf.uint8)
    R = tf.where(boolean_all, mask[..., 0], unlabeled[0])
    G = tf.where(boolean_all, mask[..., 1], unlabeled[1])
    B = tf.where(boolean_all, mask[..., 2], unlabeled[2])
    mask = tf.stack([R, G, B], axis=-1)
    return mask

def euclidean_distance(mask, color):
    sum_ = tf.cast(tf.reduce_sum(tf.math.pow(mask-color, 2), axis=2), dtype=tf.float32)
    return tf.math.sqrt(sum_)

def mask_color_correction(mask, colors):
    n_colors = tf.size(colors)//3
    mask_ = tf.cast(mask, dtype=tf.int32)
    mask_cor = tf.TensorArray(dtype=tf.float32, size=n_colors)

    for i in tf.range(n_colors):
        mask_cor = mask_cor.write(i, euclidean_distance(mask_, colors[i]))
    mask_cor = tf.transpose(mask_cor.stack(), perm=[1, 2, 0])

    mask_cor = tf.cast(tf.argmin(mask_cor, axis=2), dtype=tf.int32)
    mask_bool = tf.cast(tf.tile(tf.expand_dims(tf.clip_by_value(tf.reduce_max(mask, 2), 0, 1), -1), [1, 1, 3]), dtype=tf.uint8)
    mask_cor = tf.cast(tf.gather(colors, mask_cor), dtype=tf.uint8)
    return mask_bool * mask_cor

def resize_image(img=None, major_size=None, padding=True):
    if padding:
        image = resize_with_pad(img, method="nearest", target_height=major_size, target_width=major_size)
    else:
        img_shape = tf.shape(img)[0:2]
        ar = img_shape[1]/img_shape[0] #W/H
        image = resize(img, method="nearest", size=(major_size, major_size*ar) if img_shape[0] > img_shape[1] else (major_size/ar, major_size))
    return image


def get_patches(mask, list_colors, list_classes):
    colors, _ = tf.raw_ops.UniqueV2(x=tf.reshape(mask, (-1, 3)), axis=[0])
    colors = tf.cast(colors, dtype=tf.int32)
    patches = []
    color_idx = tf.expand_dims(tf.zeros(tf.shape(colors)[0], dtype=tf.bool), axis=-1)
    for i, color in enumerate(list_colors):
        color_idx = tf.concat([color_idx, tf.expand_dims(tf.reduce_all(colors == color, 1), axis=-1)], axis=-1)
    color_idx = tf.cast(tf.argmax(color_idx, axis=1), dtype=tf.int32).numpy()

    color_idx = color_idx[color_idx > 0]-1

    for idx_classes in color_idx:
        color = tuple(list_colors[idx_classes].numpy()/255)
        obj = list_classes[idx_classes]
        patches.append(mpatches.Patch(color=color, label=obj))
    return patches

def subplot(image, num_row, num_cols, idx, title, fontsize=8, patches=None, cmap=None, vmin=None, vmax=None):
    plt.subplot(num_row, num_cols, idx)
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    plt.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.title(title, fontsize=fontsize)
    if patches:
        plt.legend(handles=patches, fontsize=5.3, facecolor='white', labelcolor='black')
    return

def plot_sample(img, mask, df_colors, num_img=1, num_cols=2, ind=0, norm=False, color_correction=False, df_colors_conv=None, size_img=None):
    color_rgb = tf.constant(df_colors["Color_RGB"].tolist())
    classes = df_colors["Classes"].tolist()

    ind += 1
    subplot(image=img, num_row=num_img, num_cols=num_cols, idx=ind, title="Aerial image")

    ind += 1
    patches = get_patches(mask, color_rgb, classes)
    subplot(image=mask, num_row=num_img, num_cols=num_cols, idx=ind, title="Original mask", patches=patches)

    if norm == True or color_correction == True:
        color_conv = tf.constant(df_colors["Conversion"].tolist())
        baseline_classes = df_colors_conv.get_classes()
        baseline_color_rgb = tf.constant(df_colors_conv.get_colors())

    if norm == True:
        ind += 1
        mask_norm = norm_colors(mask, color_rgb, color_conv)
        patches = get_patches(mask_norm, baseline_color_rgb, baseline_classes)
        subplot(image=mask_norm, num_row=num_img, num_cols=num_cols, idx=ind, title="Re-Colorized mask", patches=patches)

    if color_correction == True:
        ind += 1
        mask_cor = mask_color_correction(mask, color_rgb)
        mask_norm = norm_colors(mask_cor, color_rgb, color_conv)
        patches = get_patches(mask_norm, baseline_color_rgb, baseline_classes)
        subplot(image=mask_norm, num_row=num_img, num_cols=num_cols, idx=ind, title="Re-Colorized mask with pixel color correction", patches=patches)
    return

def plot_samples(dataset, df_colors, df_files, num_img=1, w_disp=8, h_disp=5, norm=False, color_correction=False, df_colors_conv=None, size_img=None):
    num_cols = 2
    if norm == True:
        num_cols += 1
    if color_correction == True:
        num_cols += 1

    plt.subplots(num_img, num_cols, figsize=(w_disp, h_disp))

    for i in range(num_img):
        df_colors_ = df_colors.copy()
        ind = i*num_cols
        img, mask, abb = get_data(df_files, dataset=dataset)

        if size_img is not None:
            img = resize_image(img, size_img)
            mask = resize_image(mask, size_img)

        if abb == "SO" or abb == "OEM":
            color_rgb = tf.constant(df_colors_["Color_RGB"].tolist())
            re_color = tf.constant(df_colors_["Re-Color"].tolist())
            mask = norm_colors(mask, color_rgb, re_color)
            df_colors_["Color_RGB"] = df_colors_["Re-Color"]

        if abb == "ISPRS" or abb == "FBP":
            if abb == "FBP":
                img = img[..., ::-1]
            mask = mask[..., ::-1]

        plot_sample(img, mask, df_colors_, num_img, num_cols, ind, norm, color_correction, df_colors_conv)
    return

class myLabels():
    def __init__(self):
        self.classes = {"Building":[60, 16, 152],
                        "Environment":[132, 41, 246],
                        "Road":[110, 193, 228],
                        "Unlabeled":[0, 0, 0]}

    def get_color(self, cat):
        return self.classes[cat]

    def get_colors(self):
        return list(self.classes.values())

    def get_class(self, clr):
        for class_, color in self.classes.items():
            if clr == color:
                break
        return class_

    def get_classes(self):
        return list(self.classes.keys())

def parse(example_proto):
    feature_desc = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, feature_desc)
    image = tf.io.decode_jpeg(ex['image'], channels=3)
    mask = tf.io.decode_png(ex['mask'], channels=3)
    image.set_shape([256, 256, 3])
    mask.set_shape([256, 256, 3])
    return image, mask