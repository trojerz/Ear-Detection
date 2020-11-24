import pathlib
from PIL import Image

img_dir = 'IMG_20201120_184435.jpg'
img = Image.open(img_dir).convert('RGB')
img = img.resize((480,360))
img.save(f'ears_test_250.jpg', 'JPEG')