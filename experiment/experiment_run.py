def main():
    from pathlib import Path 
    import os
    import glob
    import json
    from ncempy.io import dm
    from skimage.util import view_as_windows
    from persim import plot_diagrams
    from ripser import lower_star_img
    from persim import PersistenceImager
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import umap    
    from scipy.stats import gaussian_kde
    from sklearn.mixture import GaussianMixture
    from scipy.stats import gaussian_kde
    import pandas as pd

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


    def split_patches(patch_size, image, stride):
        blocks = view_as_windows(image, window_shape=(patch_size, patch_size), step=stride)
        patches = blocks.reshape(-1, patch_size, patch_size)

        return patches

    def PH_patch(patched_image, scale_percentile=95):
        dgms = []
        for p in patched_image:
            dgm = lower_star_img(p)
            dgm = dgm[np.isfinite(dgm[:, 1])]

            if len(dgm) == 0:
                dgms.append(dgm)
                continue

            births = dgm[:, 0]
            deaths = dgm[:, 1]

            min_birth = births.min()
            births = births - min_birth
            deaths = deaths - min_birth

            scale = np.percentile(deaths, scale_percentile)
            if scale <= 0:
                scale = 1.0  # guard against degenerate all-zero diagrams

            births = births / scale
            deaths = deaths / scale

            dgm_norm = np.column_stack([births, deaths])
            dgms.append(dgm_norm)


        vectors = []
        for dgm in dgms:
            pimg = pimgr.transform(dgm)
            vec = pimg.ravel()
            vectors.append(vec)
        vectors = np.vstack(vectors)

        return dgms, vectors
    
    def get_positions(image_data):

        blocks = view_as_windows(image_data, window_shape=(64, 64), step=16)
        n_rows, n_cols = blocks.shape[:2]

        positions = [
            (i * 16, j * 16)
            for i in range(n_rows)
            for j in range(n_cols)
        ]

        return positions

    dataset = []

    base_dir = Path(os.environ["IMAGES_DIR"]) / "2022_02_02 5nm Ag nanoparticles on UTC/"

    for dm3_file in glob.glob(os.path.join(base_dir, "*.dm3")):

        data_dict = dm.dmReader(str(Path(dm3_file).resolve()))
        image = data_dict['data']
        patched_image = split_patches(64, image, 16)
        
        pimgr = PersistenceImager(
            pixel_size=0.02,
            birth_range=(0, 2),
        )

        dgms, vectorised_patches = PH_patch(patched_image)
        X = StandardScaler().fit_transform(vectorised_patches)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=42
        )

        embedding = reducer.fit_transform(X)

        gmm = GaussianMixture(n_components=8, covariance_type="full")
        labels = gmm.fit_predict(embedding)
        probs = gmm.predict_proba(embedding)

        positions = get_positions(image)

        cluster_densities = {}
        for label in np.unique(labels):
            indices = np.where(labels == label)[0]
            cluster_dgms = [dgms[i] for i in indices]
            kde, density_grid, extent = cluster_persistence_density(cluster_dgms)
            cluster_densities[int(label)] = {
                "density_grid": density_grid.tolist(),
                "extent": list(extent),
            }
        
        image_datapoint = {
            'image':str(dm3_file),
            'clustered_labels': labels.tolist(),
            'positions': positions,
            "densities": json.dumps(cluster_densities)
        }
        
        dataset.append(image_datapoint)

    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/project/dataset")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #tifffile.imwrite(os.path.join(OUTPUT_DIR, "image_flattened.tif"), flattened_image)
    df = pd.DataFrame(dataset)
    df.to_parquet(os.path.join(OUTPUT_DIR, "normalised_run.parquet"), index=False)
    print(f'dataset exported to {OUTPUT_DIR}')


if __name__ == "__main__":
    main()