def main():
    #import parquet and unravel
    #import images folder
    #for each row in parquet, match label to image patch
    #for each cluster, build persistence density from patches
    #save persistence density per cluster
    import pandas as pd
    import glob
    import os
    from ncempy.io import dm
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from skimage.util import view_as_windows
    import numpy as np
    from ripser import lower_star_img
    import pandas as pd
    from persim import PersistenceImager


    df = pd.read_parquet(Path(os.environ["OUTPUT_DIR"]) / "all_images_8.parquet")


    def get_positions(image_data, patch_size=64, stride=16):
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

    def label_to_image(original_image, labels, positions, patch_size=64):
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
    
    def PH_patch(patched_image):
        dgms = []
        for p in patched_image:      
            dgm = lower_star_img(p)
            dgm = dgm[np.isfinite(dgm[:,1])]
            dgms.append(dgm)
        return dgms
    
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
    
    def ripser_to_gudhi(dgms):
        gudhi_pd = []
        for dim, diag in enumerate(dgms):
            for birth, death in diag:
                gudhi_pd.append((dim, (float(birth), float(death))))
        return gudhi_pd
    
    dataset = []
    
    for id, row in df.iterrows():

        image_name = Path(row["image"]).stem

        matches = list((Path(os.environ["IMAGES_DIR"])/ "2022_02_02 5nm Ag nanoparticles on UTC/").glob(f"{image_name}.dm3"))

        if not matches:
            continue

        image_path = matches[0]
        data_dict = dm.dmReader(str(Path(image_path).resolve()))
        image = data_dict['data']

        labels = row["clustered"]
        dgms = []
        for patch_batch in iter_patch_batches(image):
            dgms.extend(PH_patch(patch_batch))  
        
        pim = PersistenceImager(
        pixel_size=16,
        birth_range=(400, 800),
        pers_range=(0, 300)
    )
        pim.fit(dgms)

        clusters = [int(i) for i in np.unique(labels)]
        cluster_dgms = {i: [] for i in clusters}

        for dgm, label in zip(dgms, labels):
            cluster_dgms[label].append(dgm)

        mean_densities = {}
        for i in clusters:
            if not cluster_dgms[i]:
                mean_densities[str(i)] = None
                continue
            imgs = pim.transform(cluster_dgms[i])
            mean_densities[str(i)] = np.mean(imgs, axis=0).tolist()

        image_datapoint = {
            'image':str(image_name),
            'cluster_summaries': mean_densities
        }
        
        dataset.append(image_datapoint)

    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/project/dataset")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(dataset)
    df.to_parquet(os.path.join(OUTPUT_DIR, "density_analysis.parquet"), index=False)
    print(f'dataset exported to {OUTPUT_DIR}')
        

if __name__ == "__main__":
    main()