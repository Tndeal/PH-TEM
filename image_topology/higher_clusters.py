def main():

    import glob
    import os
    from ncempy.io import dm
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from sklearn.preprocessing import StandardScaler
    import umap
    from sklearn.cluster import KMeans
    from skimage.util import view_as_windows
    import numpy as np
    from persim import PersistenceImager
    from ripser import lower_star_img
    import pandas as pd
    from itertools import islice

    def split_patches(patch_size, image, stride):
        blocks = view_as_windows(image, window_shape=(patch_size, patch_size), step=stride)
        patches = blocks.reshape(-1, patch_size, patch_size)

        return patches

    pimgr = PersistenceImager(
        pixel_size=16,
        birth_range=(400, 700),
        pers_range=(0, 300)
    )

    def PH_patch(patched_image):
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

    def get_positions(image_data, patch_size=16, stride=None):
        if stride is None:
            stride = patch_size

        blocks = view_as_windows(
            image_data,
            (patch_size, patch_size),
            step=stride
        )

        n_rows, n_cols = blocks.shape[:2]

        positions = [
            (i * stride, j * stride)
            for i in range(n_rows)
            for j in range(n_cols)
        ]

        return positions

    def label_to_image(original_image, labels, positions, patch_size=16, n_clusters=None):
        H, W = original_image.shape[:2]

        if n_clusters is None:
            n_clusters = np.max(labels) + 1

        votes = np.zeros((H, W, n_clusters), dtype=np.float32)

        for (r, c), lab in zip(positions, labels):
            votes[r:r+patch_size, c:c+patch_size, lab] += 1

        return np.argmax(votes, axis=-1)


    analysis_dataset = []

    base_dir = Path(os.environ["IMAGES_DIR"]) / "2022_02_02 5nm Ag nanoparticles on UTC"

    for dm3_file in islice(base_dir.glob("*.dm3"), 1):
        truth_image = (base_dir
            / "Labels"
            / f"{dm3_file.stem}_label.png"
        )
        truth_image = mpimg.imread(truth_image)

        print("Processing:", dm3_file)

        data_dict = dm.dmReader(str(Path(dm3_file).resolve()))
        image = data_dict['data']
        vectorised_patches = PH_patch(split_patches(16, image, 8))

        X = StandardScaler().fit_transform(vectorised_patches)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=42
        )

        embedding = reducer.fit_transform(X)

        kmeans = KMeans(n_clusters=8, random_state=42)
        labels = kmeans.fit_predict(embedding)

        positions = get_positions(image)
        label_img = label_to_image(image, labels, positions, 8)

        mean_intensities = [image[label_img==cl].mean() for cl in np.unique(labels)]
        crystal_cluster = np.argmin(mean_intensities)

        mask = (label_img == crystal_cluster).astype(np.uint8)

        image_datapoint = {
            'name': dm3_file,
            'vectors': vectorised_patches,
            'labels': labels,
            'cluster_image': label_img,
        }
        analysis_dataset.append(image_datapoint)

    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/project/dataset")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.DataFrame(analysis_dataset)
    df.to_parquet(os.path.join(OUTPUT_DIR, "analysis_dataset.parquet"), index=False)
    print(f'dataset exported to {OUTPUT_DIR}')

if __name__ == "__main__":
    main()