    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:

def main():
    from skimage.util import view_as_blocks
    print('started')

    def split_patches(patch_size, image):
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size))
        patches = blocks.reshape(-1, patch_size, patch_size)

        return patches


    # In[17]:


    from persim import plot_diagrams
    from ripser import lower_star_img
    from persim import PersistenceImager
    import numpy as np
    import pandas as pd

    pimgr = PersistenceImager(
        pixel_size=16,
        birth_range=(400, 700),
        pers_range=(0, 300)
    )

    def PH_patch(patched_image, pimgr = pimgr):
        dgms = []
        for p in patched_image:      
            dgm = lower_star_img(p)
            dgm = dgm[np.isfinite(dgm[:,1])]
            dgms.append(dgm)
        pimgr.fit(dgms)
        vectors = []
        for dgm in dgms:
            pimg = pimgr.transform(dgm)
            vec = pimg.ravel()
            vectors.append(vec)
        vectors = np.vstack(vectors)
        return vectors


    # In[3]:


    def get_positions(image_data):

        blocks = view_as_blocks(image_data, block_shape=(16, 16))
        n_rows, n_cols = blocks.shape[:2]

        positions = [
            (i * 16, j * 16)
            for i in range(n_rows)
            for j in range(n_cols)
        ]

        return positions


    # In[4]:


    import numpy as np

    def label_to_image(original_image, labels, positions):

        H, W = original_image.shape[:2]
        label_img = np.zeros((H, W), dtype=int)

        for (r, c), lab in zip(positions, labels):
            label_img[r:r+16, c:c+16] = lab

        return label_img


    # In[32]:


    import glob
    import os
    from ncempy.io import dm
    from sklearn.preprocessing import StandardScaler
    import umap
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from pathlib import Path
    import matplotlib.image as mpimg

    IMAGES_DIR = os.environ.get("IMAGES_DIR", "./images")

    analysis_dataset = []

    for dm3_file in glob.glob(os.path.join(IMAGES_DIR+"/2022_02_02 5nm Ag nanoparticles on UTC/", "*.dm3")):
        print('file')
        file = Path(dm3_file)
        data_dict = dm.dmReader(str(file.resolve()))

        image_data = data_dict['data']

        truth_image = dm3_file.replace("/2022_02_02 5nm Ag nanoparticles on UTC/", "/2022_02_02 5nm Ag nanoparticles on UTC/Labels/").replace(".dm3", "_label.png")
        truth_image = mpimg.imread(truth_image)

        image_datapoint = {
            'original': image_data,
            'truth': truth_image,
            'vectors': [],
            'labels': []
        }
        analysis_dataset.append(image_datapoint)


    # In[33]:


    import numpy as np


    def patch_label(truth_patch, threshold=0.4):
        fraction = np.sum(truth_patch) / truth_patch.size
        return int(fraction > threshold)


    # In[ ]:


    for img_object in analysis_dataset:
        patched_image = split_patches(16, img_object['original'])
        vectorised_patches = PH_patch(patched_image)
        img_object['vectors'] = vectorised_patches

        truth_patches = split_patches(16, img_object['truth'])
        labels = []
        for t in truth_patches:
            label = patch_label(t)
            labels.append(label)

        img_object['labels'] = labels


    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/project/dataset")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.DataFrame(analysis_dataset)
    df.to_parquet(os.path.join(OUTPUT_DIR, "analysis_dataset.parquet"), index=False)
    print(f'dataset exported to {OUTPUT_DIR}')

if __name__ == "__main__":
    main()