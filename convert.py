import numpy as np
from PIL import Image
import os
import struct

def read_idx(filename):
    with open(filename, 'rb') as f:
        # Read the magic number
        magic = struct.unpack('>I', f.read(4))[0]
        if magic == 2051:  # Images
            # Read dimensions
            num_images = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            # Read image data
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
            return images
        elif magic == 2049:  # Labels
            num_labels = struct.unpack('>I', f.read(4))[0]
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
        else:
            raise ValueError("Invalid IDX file")

def save_images_as_jpg(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img_pil = Image.fromarray(img)
        img_pil = img_pil.convert("L")  # Convert to grayscale
        img_pil.save(os.path.join(output_dir, f'image_{i}.png'))

# Chemins des fichiers
test_images_path = 'data/MNIST/MNIST/raw/t10k-images-idx3-ubyte'

# Lecture des images
test_images = read_idx(test_images_path)

# Sauvegarde des images en format JPG
save_images_as_jpg(test_images, 'test_images')
