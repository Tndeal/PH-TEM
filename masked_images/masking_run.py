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
    from scipy.stats import gaussian_kde
    import json

    pimgr = PersistenceImager(
        pixel_size=16,
        birth_range=(400, 800),
        pers_range=(0, 300)
    )
    pimgr_fit = False

    def cluster_persistence_density(dgms_in_cluster, grid_size=100, bw_method=None):
        all_points = np.vstack([d for d in dgms_in_cluster if len(d) > 0])
        births = all_points[:, 0]
        lifetimes = all_points[:, 1] - all_points[:, 0]

        coords = np.vstack([births, lifetimes])
        kde = gaussian_kde(coords, bw_method=bw_method)

        # evaluate on a grid for visualisation
        b_min, b_max = births.min(), births.max()
        l_min, l_max = lifetimes.min(), lifetimes.max()

        bb, ll = np.meshgrid(
            np.linspace(b_min, b_max, grid_size),
            np.linspace(l_min, l_max, grid_size)
        )
        grid_coords = np.vstack([bb.ravel(), ll.ravel()])
        density_grid = kde(grid_coords).reshape(grid_size, grid_size)

        return kde, density_grid, (b_min, b_max, l_min, l_max)

    def PH_patch(patched_image):
        nonlocal pimgr_fit
        dgms = []
        for p in patched_image:      
            dgm = lower_star_img(p)
            dgm = dgm[np.isfinite(dgm[:,1])]
            dgms.append(dgm)
        if not pimgr_fit:
            for dgm in dgms:
                if len(dgm) > 0 and np.isfinite(dgm).all():
                    pimgr.fit(dgm)
                    pimgr_fit = True
                    break
        vectors = []
        for dgm in dgms:
            if len(dgm) == 0 or not pimgr_fit:
                vectors.append(np.zeros(pimgr.resolution[0] * pimgr.resolution[1]))
            else:
                pimg = pimgr.transform(dgm)
                vec = pimg.ravel()
                vectors.append(vec)
        vectors = np.vstack(vectors)
        return dgms, vectors

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

    dataset = []

    base_dir = Path("../hrtem_files") / "2022_02_02 5nm Ag nanoparticles on UTC/"

    for i, dm3_file in enumerate(glob.glob(os.path.join(base_dir, "*.dm3"))):
        if i >= 1:
            break
        data_dict = dm.dmReader(str(Path(dm3_file).resolve()))
        image = data_dict['data']
        
        truth_image = dm3_file.replace("/2022_02_02 5nm Ag nanoparticles on UTC/", "/2022_02_02 5nm Ag nanoparticles on UTC/Labels/").replace(".dm3", "_label.png")
        truth_image = mpimg.imread(truth_image)
        blurred = gaussian_filter(truth_image, sigma=20)
        mask = blurred > 0
    
        masked_image = np.where(mask, image, np.nan)

        all_vectors = []
        all_dgms = []

        for patch_batch in iter_patch_batches(masked_image):
            #normalized_batch = [z_score_transform(p) for p in patch_batch]
            dgms, vec = PH_patch(patch_batch)
            all_vectors.append(vec)
            all_dgms.extend(dgms)

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

        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=4, covariance_type="full")
        labels = gmm.fit_predict(embedding)

        positions = get_positions(masked_image, 64, 16)
        #label_img = label_to_image(image, labels, positions, 16)

        #mean_intensities = [image[label_img==cl].mean() for cl in np.unique(labels)]
        #crystal_cluster = np.argmin(mean_intensities)
        #mask = (label_img == crystal_cluster).astype(np.uint8)

        cluster_densities = {}
        for label in np.unique(labels):
            indices = np.where(labels == label)[0]
            cluster_dgms = [all_dgms[i] for i in indices]
            kde, density_grid, extent = cluster_persistence_density(cluster_dgms)
            cluster_densities[int(label)] = {
                "density_grid": density_grid.tolist(),
                "extent": list(extent),
            }
        
        image_datapoint = {
            'image':str(dm3_file),
            'clustered_labels': labels,
            'positions': positions,
            "densities": json.dumps(cluster_densities)
        }
        dataset.append(image_datapoint)

    OUTPUT_DIR = "./output/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #tifffile.imwrite(os.path.join(OUTPUT_DIR, "image_flattened.tif"), flattened_image)
    df = pd.DataFrame(dataset)
    df.to_parquet(os.path.join(OUTPUT_DIR, "masked_images_densities.parquet"), index=False)
    print(f'dataset exported to {OUTPUT_DIR}')

if __name__ == "__main__":
    main()