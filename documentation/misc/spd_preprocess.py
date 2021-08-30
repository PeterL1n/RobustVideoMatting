# pip install supervisely
import supervisely_lib as sly
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# Download dataset from <https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets>
project_root = 'PATH_TO/Supervisely Person Dataset'  # <-- Configure input
project = sly.Project(project_root, sly.OpenMode.READ)

output_path = 'OUTPUT_DIR'  # <-- Configure output
os.makedirs(os.path.join(output_path, 'train', 'src'))
os.makedirs(os.path.join(output_path, 'train', 'msk'))
os.makedirs(os.path.join(output_path, 'valid', 'src'))
os.makedirs(os.path.join(output_path, 'valid', 'msk'))

max_size = 2048  # <-- Configure max size

for dataset in project.datasets:
    for item in tqdm(dataset):
        ann = sly.Annotation.load_json_file(dataset.get_ann_path(item), project.meta)
        msk = np.zeros(ann.img_size, dtype=np.uint8)
        for label in ann.labels:
            label.geometry.draw(msk, color=[255])
        msk = Image.fromarray(msk)
        
        img = Image.open(dataset.get_img_path(item)).convert('RGB')
        if img.size[0] > max_size or img.size[1] > max_size:
            scale = max_size / max(img.size)
            img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.BILINEAR)
            msk = msk.resize((int(msk.size[0] * scale), int(msk.size[1] * scale)), Image.NEAREST)
        
        img.save(os.path.join(output_path, 'train', 'src', item.replace('.png', '.jpg')))
        msk.save(os.path.join(output_path, 'train', 'msk', item.replace('.png', '.jpg')))

# Move first 100 to validation set
names = os.listdir(os.path.join(output_path, 'train', 'src'))
for name in tqdm(names[:100]):
    os.rename(
        os.path.join(output_path, 'train', 'src', name),
        os.path.join(output_path, 'valid', 'src', name))
    os.rename(
        os.path.join(output_path, 'train', 'msk', name),
        os.path.join(output_path, 'valid', 'msk', name))