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
    from skimage.transform import resize
    from scipy.ndimage import gaussian_filter
    import numpy as np
    from persim import PersistenceImager
    from ripser import lower_star_img
    import pandas as pd
    from itertools import islice
    import tifffile
    from skimage.restoration import denoise_nl_means

    def z_score_transform(patch):
        std = np.std(patch)
        if std > 1e-6:
            return (patch - np.mean(patch)) / std
        return patch - np.mean(patch)

    def split_patches(patch_size, image, stride):
        blocks = view_as_windows(image, window_shape=(patch_size, patch_size), step=stride)
        patches = blocks.reshape(-1, patch_size, patch_size)

        return patches

    pimgr = PersistenceImager(
        pixel_size=16,
        birth_range=(400, 800),
        pers_range=(0, 300)
    )
    pimgr_fit = False

    def PH_patch(patched_image):
        nonlocal pimgr_fit
        dgms = []
        for p in patched_image:      
            dgm = lower_star_img(p)
            dgm = dgm[np.isfinite(dgm[:,1])]
            dgms.append(dgm)
        if pimgr_fit == False:
            pimgr.fit(dgms[0])
            pimgr_fit = True
        vectors = []
        for dgm in dgms:
            if len(dgm) == 0:
                vectors.append(np.zeros(pimgr.resolution[0] * pimgr.resolution[1]))
            else:
                pimg = pimgr.transform(dgm)
                vec = pimg.ravel()
                vectors.append(vec)
        vectors = np.vstack(vectors)
        return vectors

    def get_positions(image_data, patch_size=64, stride=None):
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

    def label_to_image(original_image, labels, positions, patch_size=16):
        H, W = original_image.shape[:2]

        label_img = np.zeros((H, W), dtype=np.uint8)
        vote_img  = np.zeros((H, W), dtype=np.uint8)

        for (r, c), lab in zip(positions, labels):
            r_end = min(r + patch_size, H)
            c_end = min(c + patch_size, W)

            region = (slice(r, r_end), slice(c, c_end))

            votes = vote_img[region] + 1

            mask = votes >= vote_img[region]

            label_patch = label_img[region]
            label_patch[mask] = lab

            vote_img[region][mask] = votes[mask]

        return label_img

    def iter_patch_batches(image, patch_size=64, stride=16, batch_size=500):
        patches = []

        blocks = view_as_windows(image, (patch_size, patch_size), step=stride)
        n_rows, n_cols = blocks.shape[:2]

        for i in range(n_rows):
            for j in range(n_cols):
                patches.append(blocks[i, j])

                if len(patches) == batch_size:
                    yield np.array(patches)
                    patches = []

        if patches:
            yield np.array(patches)

    analysis_dataset = []

    base_dir = Path(os.environ["IMAGES_DIR"]) / "2022_02_02 5nm Ag nanoparticles on UTC"

    image = base_dir / "20220202_Ag_UTC_330kx_2650e_0p1596s_05.dm3"
    truth_image = (base_dir
        / "Labels"
        / "20220202_Ag_UTC_330kx_2650e_0p1596s_05_label.png"
    )
    truth_image = mpimg.imread(truth_image)


    data_dict = dm.dmReader(str(Path(image).resolve()))
    image = data_dict['data']
    

    #image_small = resize(image, (256, 256), anti_aliasing=True).astype(np.float32)
    
    #field = gaussian_filter(image, sigma=500)
    #large_scale_field = resize(field_small, image.shape, order=1, mode='reflect').astype(np.float32)

    #global_mean = np.mean(image)
    #flattened_image = (image / (field + 1e-6)) * global_mean
    #denoised = denoise_nl_means(flattened_image, h=0.02)
    #flattened_image = np.clip(flattened_image, 0, 65535).astype(np.uint16)
    all_vectors = []

    for patch_batch in iter_patch_batches(image):
        #normalized_batch = [z_score_transform(p) for p in patch_batch]
        vec = PH_patch(patch_batch)
        all_vectors.append(vec)

    vectorised_patches = np.vstack(all_vectors)

    X = StandardScaler().fit_transform(vectorised_patches)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        low_memory=True
    )

    embedding = reducer.fit_transform(X)

    kmeans = KMeans(n_clusters=7, random_state=42)
    labels = kmeans.fit_predict(embedding)

    #positions = get_positions(image, 8)
    #label_img = label_to_image(image, labels, positions)

    #mean_intensities = [image[label_img==cl].mean() for cl in np.unique(labels)]
    #crystal_cluster = np.argmin(mean_intensities)

    #mask = (label_img == crystal_cluster).astype(np.uint8)

    image_datapoint = {
        'clustered': labels,
    }

    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/project/dataset")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #tifffile.imwrite(os.path.join(OUTPUT_DIR, "image_flattened.tif"), flattened_image)
    df = pd.DataFrame(image_datapoint)
    df.to_parquet(os.path.join(OUTPUT_DIR, "larger_patch.parquet"), index=False)
    print(f'dataset exported to {OUTPUT_DIR}')

if __name__ == "__main__":
    main()