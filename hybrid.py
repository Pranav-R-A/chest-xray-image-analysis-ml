import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from collections import Counter
import csv
import hashlib
from joblib import Parallel, delayed

"""
modified.py

Tools to run unsupervised analysis on the chest_xray dataset.

Features/flows included:
- Load images from train/val/test (folder structure used by your project)
- Extract simple features (downsampled pixels or HOG)
- Run KMeans clustering (n_clusters=2 by default)
- Visualize clusters with PCA and t-SNE and show sample images per cluster
- Optional: simple convolutional autoencoder for anomaly detection (requires TensorFlow)

Usage:
    python unsupervised.py

Adjust parameters at the top of the file.
"""

# --- User parameters ---
BASE_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

IMG_SIZE = (128, 128)   # size to which images are resized for feature extraction
# Set to -1 for no limit (use with caution, may be slow)
MAX_PER_CLASS = 1000      # default limit images per class to keep runs fast
FEATURE_METHOD = 'hog'  # 'pixels' or 'hog' or 'multilayer' or 'efnet' (efnet uses pretrained CNN embeddings)
N_CLUSTERS = 2
# Runtime flags (can be overridden from CLI)
PREPROCESS = True
NORMALIZE = False
PURITY_THRESHOLD = 0.6
UNCERT_DIST_MULT = 2.0
NO_TSNE = False
CACHE_FEATURES = False
N_JOBS = 1
# Region weighting defaults (center/lungs/peripheral/full)
CENTER_WEIGHT = 2.0
LUNG_WEIGHT = 1.5
PERIPHERAL_WEIGHT = 0.8
FULL_WEIGHT = 1.0


def list_images(folder, max_per_class=MAX_PER_CLASS):
    images = []
    labels = []
    paths = []
    # expect subfolders like NORMAL/ and PNEUMONIA/
    for cls in sorted(os.listdir(folder)):
        cls_dir = os.path.join(folder, cls)
        if not os.path.isdir(cls_dir):
            continue
        count = 0
        for fname in sorted(os.listdir(cls_dir)):
            # handle unlimited mode
            if max_per_class != -1 and count >= max_per_class:
                break
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                p = os.path.join(cls_dir, fname)
                images.append(p)
                # normalize detailed labels: treat bacterial/viral as PNEUMONIA (collapse to two classes)
                if cls.upper() == 'PNEUMONIA':
                    labels.append('PNEUMONIA')
                else:
                    labels.append(cls.upper())
                paths.append(p)
                count += 1
    return images, labels, paths


def load_and_preprocess(path, img_size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f'Could not load {path}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply optional preprocessing and normalization controlled by module flags
    if PREPROCESS:
        gray = preprocess_gray(gray)
    resized = cv2.resize(gray, img_size)
    if NORMALIZE:
        resized = (resized.astype('float32') - resized.min()) / (resized.max() - resized.min() + 1e-8)
    return resized


def preprocess_gray(img_gray, do_clahe=True, do_denoise=True):
    """Apply CLAHE and denoising to reduce nuisance variation."""
    out = img_gray
    if do_clahe:
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            out = clahe.apply(out)
        except Exception:
            out = out
    if do_denoise:
        try:
            out = cv2.fastNlMeansDenoising(out, None, 10, 7, 21)
        except Exception:
            out = out
    return out


def image_quality_metrics(img_gray):
    """Compute a small set of image-level quality/opacity metrics.
    Returns a dict with keys: snr, lap_var, frac_bright, entropy.
    """
    res = {}
    a = img_gray.astype(np.float32)
    mean = float(a.mean())
    std = float(a.std()) + 1e-8
    res['snr'] = mean / std
    # Laplacian variance -> blur detector
    try:
        res['lap_var'] = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    except Exception:
        res['lap_var'] = 0.0
    # fraction of bright pixels (opacity proxy)
    res['frac_bright'] = float((img_gray > np.percentile(img_gray, 75)).sum()) / img_gray.size
    # entropy
    hist = cv2.calcHist([img_gray.astype('uint8')], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-12)
    res['entropy'] = float(-np.sum(hist * np.log2(hist + 1e-12)))
    return res


def hog_features(img_gray):
    # lazy import to avoid hard dependency when not needed
    from skimage.feature import hog
    feat = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return feat


def _split_regions_and_resize(img_gray, out_size=(64, 64)):
    """
    Split a grayscale image into multiple regions that approximate multi-layer
    encodings (full image, center, left/right lung regions, upper/lower halves).
    Each region is resized to `out_size` so HOG descriptors have a consistent
    length across regions.
    Returns a list of (name, region_image) pairs.
    """
    h, w = img_gray.shape
    regions = []

    # full
    regions.append(('full', cv2.resize(img_gray, out_size)))

    # center vertical band (likely to contain lungs)
    cx1 = int(w * 0.25)
    cx2 = int(w * 0.75)
    cy1 = int(h * 0.1)
    cy2 = int(h * 0.9)
    ctr = img_gray[cy1:cy2, cx1:cx2]
    if ctr.size == 0:
        ctr = img_gray
    regions.append(('center', cv2.resize(ctr, out_size)))

    # left lung region (left half, centered vertically)
    lx1 = 0
    lx2 = int(w * 0.5)
    ly1 = int(h * 0.1)
    ly2 = int(h * 0.9)
    left = img_gray[ly1:ly2, lx1:lx2]
    if left.size == 0:
        left = img_gray
    regions.append(('left_lung', cv2.resize(left, out_size)))

    # right lung region
    rx1 = int(w * 0.5)
    rx2 = w
    right = img_gray[ly1:ly2, rx1:rx2]
    if right.size == 0:
        right = img_gray
    regions.append(('right_lung', cv2.resize(right, out_size)))

    # upper half and lower half
    upper = img_gray[0:int(h * 0.5), :]
    lower = img_gray[int(h * 0.5):h, :]
    if upper.size == 0:
        upper = img_gray
    if lower.size == 0:
        lower = img_gray
    regions.append(('upper', cv2.resize(upper, out_size)))
    regions.append(('lower', cv2.resize(lower, out_size)))

    return regions


def extract_multilayer_features(paths):
    """
    For each image path, split the image into multiple regions and for each
    region compute HOG features plus a few simple intensity statistics. The
    region HOGs and stats are concatenated into a single vector per image.

    This creates a multi-layer encoding that captures both global and
    localized textural/opacity information (useful to capture patterns such
    as concentrated vs. diffuse opacities in lung regions).
    """
    feats = []
    for p in paths:
        img = load_and_preprocess(p)
        regions = _split_regions_and_resize(img, out_size=(64, 64))
        img_feats = []
        for name, reg in regions:
            # compute simple stats
            mean = float(np.mean(reg))
            std = float(np.std(reg))
            # fraction of pixels above a low threshold to capture opacity coverage
            frac_non_dark = float((reg > 10).sum()) / (reg.size + 1e-12)
            # HOG descriptor on the region
            hogf = hog_features(reg)
            # append stats then hog
            img_feats.append(mean)
            img_feats.append(std)
            img_feats.append(frac_non_dark)
            img_feats.extend(hogf.tolist())
        feats.append(np.array(img_feats, dtype=np.float32))
    # append image-level quality metrics to each feature vector
    feats = np.array(feats)
    q_feats = []
    for p in paths:
        img = load_and_preprocess(p)
        qm = image_quality_metrics(img)
        q_feats.append([qm['snr'], qm['lap_var'], qm['frac_bright'], qm['entropy']])
    q_feats = np.array(q_feats, dtype=np.float32)
    feats = np.hstack([feats, q_feats])
    return feats


def extract_features(paths, method=FEATURE_METHOD):
    feats = []
    # multilayer: region-based HOG + stats encoding
    if method in ('multilayer', 'multilayer_hog'):
        return extract_multilayer_features(paths)
    if method == 'efnet':
        # Use CNN embeddings (EfficientNet / MobileNet fallback)
        model_name = globals().get('CNN_MODEL', 'efficientnet')
        batch_size = globals().get('CNN_BATCH_SIZE', 32)
        return extract_cnn_embeddings(paths, model_name=model_name, batch_size=batch_size, cache=CACHE_FEATURES)

    # feature caching support: compute a cache key from file list and method
    if CACHE_FEATURES:
        h = hashlib.sha1()
        h.update(method.encode('utf-8'))
        for p in paths:
            h.update(p.encode('utf-8'))
        cache_fname = os.path.join(BASE_DIR, f'feature_cache_{h.hexdigest()}.npz')
        if os.path.exists(cache_fname):
            try:
                data = np.load(cache_fname, allow_pickle=True)
                X = data['X']
                return X
            except Exception:
                pass

    def process_path(p):
        img = load_and_preprocess(p)
        if method == 'pixels':
            base = img.flatten()
            qm = image_quality_metrics(img)
            return np.concatenate([base, np.array([qm['snr'], qm['lap_var'], qm['frac_bright'], qm['entropy']])])
        elif method == 'hog':
            base = hog_features(img)
            qm = image_quality_metrics(img)
            return np.concatenate([base, np.array([qm['snr'], qm['lap_var'], qm['frac_bright'], qm['entropy']])])
        else:
            raise ValueError('Unknown feature method')

    if N_JOBS == 1:
        feats_list = [process_path(p) for p in paths]
    else:
        feats_list = Parallel(n_jobs=N_JOBS)(delayed(process_path)(p) for p in paths)

    feats = np.vstack(feats_list) if len(feats_list) > 0 else np.empty((0, 0), dtype=np.float32)

    if CACHE_FEATURES:
        try:
            np.savez_compressed(cache_fname, X=feats)
            print(f'Saved features to cache {cache_fname}')
        except Exception:
            pass

    return feats


def cluster_and_visualize(X, images_paths, labels=None, n_clusters=N_CLUSTERS, clusterer='kmeans', prob_threshold=0.0, map_threshold=0.5):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA for dimensionality reduction before clustering (optional)
    pca = PCA(n_components=min(50, Xs.shape[1]))
    Xp = pca.fit_transform(Xs)

    clusters = None
    if clusterer == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = model.fit_predict(Xp)
    elif clusterer == 'gmm':
        gmm = GaussianMixture(n_components=n_clusters, reg_covar=1e-3, random_state=42)
        gmm.fit(Xp)
        probs = gmm.predict_proba(Xp)
        max_probs = probs.max(axis=1)
        clusters = probs.argmax(axis=1)
        if prob_threshold > 0:
            clusters = np.where(max_probs >= prob_threshold, clusters, -1)
    elif clusterer == 'agglo':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = model.fit_predict(Xp)
    else:
        raise ValueError(f'Unknown clusterer: {clusterer}')

    # Print counts for each cluster (includes ambiguous '-1' if present)
    cluster_counts = Counter(clusters)
    print('Cluster counts:', {int(k) if isinstance(k, (np.integer, int)) else k: int(v) for k, v in cluster_counts.items()})

    # Save cluster assignments
    out_csv = os.path.join(BASE_DIR, 'cluster_assignments.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_label' if labels is not None else 'unknown', 'cluster'])
        for p, lab, c in zip(images_paths, labels or ['']*len(images_paths), clusters):
            writer.writerow([p, lab, int(c) if isinstance(c, (int, np.integer)) else c])
    print(f'Wrote cluster assignments to {out_csv}')

    # If true labels were provided, compute clustering metrics
    if labels is not None and len(labels) == len(clusters):
        # Convert string labels to integers for metrics
        unique_labels = sorted(list(set(labels)))
        label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
        y_true = np.array([label_to_int[l] for l in labels])
        ari = adjusted_rand_score(y_true, clusters)
        nmi = normalized_mutual_info_score(y_true, clusters)
        print(f'Adjusted Rand Index: {ari:.4f}')
        print(f'Normalized Mutual Information: {nmi:.4f}')

        # Determine mapping from clusters -> most frequent detailed label (e.g., NORMAL/PNEUMONIA)
        mapping = {}
        cluster_label_counts = {}
        for c in np.unique(clusters):
            idxs = np.where(clusters == c)[0]
            if len(idxs) == 0:
                continue
            assigned = [labels[i] for i in idxs]
            cnt = Counter(assigned)
            most_common_label, most_common_count = cnt.most_common(1)[0]
            purity = most_common_count / len(idxs)
            if purity >= PURITY_THRESHOLD:
                mapping[c] = most_common_label
            else:
                mapping[c] = 'UNKNOWN'
            # store full counts for diagnostics
            cluster_label_counts[int(c)] = dict(cnt)

        # compute mapped accuracy
        mapped_preds = [mapping.get(c, None) for c in clusters]
        valid_idx = [i for i, mp in enumerate(mapped_preds) if mp is not None and clusters[i] != -1]
        if len(valid_idx) > 0:
            mapped_correct = sum(1 for i in valid_idx if labels[i] == mapped_preds[i])
            mapped_acc = mapped_correct / len(valid_idx)
        else:
            mapped_acc = 0.0
        mapped_correct_all = sum(1 for gt, mp in zip(labels, mapped_preds) if mp is not None and gt == mp)
        mapped_acc_all = mapped_correct_all / len(labels)
        print(f'Cluster -> label mapping (majority vote): {mapping}')
        print(f'Per-cluster label counts: {cluster_label_counts}')
        print(f'Mapped cluster accuracy (ignoring ambiguous): {mapped_acc:.4f}')
        print(f'Mapped cluster accuracy (ambiguous counted wrong): {mapped_acc_all:.4f}')
        print('Confusion matrix (rows=true labels, cols=clusters):')
        cm = confusion_matrix(y_true, clusters)
        print(cm)

    # PCA plot (2D)
    pca2 = PCA(n_components=2)
    X2 = pca2.fit_transform(Xp)
    plt.figure(figsize=(6,6))
    # iterate over actual cluster ids (may include -1 for ambiguous assignments)
    for c in np.unique(clusters):
        idx = clusters == c
        plt.scatter(X2[idx,0], X2[idx,1], label=f'cluster {c}', alpha=0.6)
    plt.legend()
    plt.title('PCA 2D view of clusters')
    plt.tight_layout()
    plt.show()

    # t-SNE plot (slower) -- skip if requested
    if not NO_TSNE:
        try:
            tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
            Xt = tsne.fit_transform(Xp)
            plt.figure(figsize=(6,6))
            for c in np.unique(clusters):
                idx = clusters == c
                plt.scatter(Xt[idx,0], Xt[idx,1], label=f'cluster {c}', alpha=0.6)
            plt.legend()
            plt.title('t-SNE view of clusters')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print('t-SNE failed or is slow on your machine:', e)

    # show sample images from each cluster
    samples_per_cluster = 6
    for c in np.unique(clusters):
        idxs = np.where(clusters == c)[0]
        if len(idxs) == 0:
            continue
        pick = idxs[:samples_per_cluster]
        plt.figure(figsize=(12, 4))
        for i, pi in enumerate(pick):
            img = cv2.imread(images_paths[pi])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, samples_per_cluster, i+1)
            plt.imshow(img)
            title = os.path.basename(images_paths[pi])
            if labels is not None:
                title = f'{labels[pi]}\n{title}'
            plt.title(title, fontsize=8)
            plt.axis('off')

def extract_cnn_embeddings(paths, model_name='efficientnet', batch_size=32, cache=True):
    """Extract CNN embeddings using TensorFlow EfficientNet or PyTorch MobileNet fallback.
    Returns a (N, D) numpy array of embeddings. Caches embeddings to disk when `cache` is True.
    """
    if cache:
        h = hashlib.sha1()
        h.update(model_name.encode('utf-8'))
        for p in paths:
            h.update(p.encode('utf-8'))
        cache_file = os.path.join(BASE_DIR, f'cnn_emb_{model_name}_{h.hexdigest()}.npz')
        if os.path.exists(cache_file):
            try:
                data = np.load(cache_file)
                print(f'Loaded cached CNN embeddings from {cache_file}')
                return data['X']
            except Exception:
                pass

    # Try TensorFlow EfficientNet first
    use_tf = False
    use_pt = False
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
        model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        use_tf = True
        emb_size = model.output_shape[-1]
        def preprocess_batch(imgs):
            arr = np.array(imgs, dtype=np.float32)
            return preprocess_input(arr)
        def run_batch(batch_imgs):
            arr = preprocess_batch(batch_imgs)
            return model.predict(arr, verbose=0)
    except Exception:
        use_tf = False

    # If TF not available, try PyTorch MobileNetV2
    if not use_tf:
        try:
            import torch
            import torchvision.transforms as T
            import torchvision.models as models
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pt_model = models.mobilenet_v2(pretrained=True)
            # Replace classifier with identity to get features
            try:
                pt_model.classifier = torch.nn.Identity()
            except Exception:
                pass
            pt_model.eval()
            pt_model.to(device)
            preprocess = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            emb_size = 1280 if hasattr(pt_model, 'last_channel') else 1024
            def run_batch(batch_imgs):
                tensors = [preprocess(img) for img in batch_imgs]
                arr = torch.stack(tensors).to(device)
                with torch.no_grad():
                    out = pt_model(arr)
                return out.cpu().numpy()
            use_pt = True
        except Exception:
            use_pt = False

    if not use_tf and not use_pt:
        raise RuntimeError('No supported CNN backend available (TensorFlow or PyTorch required)')

    embs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                img = cv2.resize(img, (224, 224))
            except Exception:
                # fallback: skip images that cannot be resized
                continue
            imgs.append(img)
        if len(imgs) == 0:
            continue
        emb = run_batch(imgs)
        embs.append(emb)

    if len(embs) == 0:
        return np.empty((0, emb_size))
    feats = np.vstack(embs)
    if cache:
        try:
            np.savez_compressed(cache_file, X=feats)
            print(f'Saved CNN embeddings to {cache_file}')
        except Exception:
            pass
    return feats
    inp = layers.Input(inp_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(8, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D(2, padding='same')(x)

    x = layers.Conv2D(8, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    ae = models.Model(inp, decoded)
    ae.compile(optimizer='adam', loss='mse')
    print('Training autoencoder on', Xn.shape[0], 'normal images...')
    ae.fit(Xn, Xn, epochs=epochs, batch_size=8, validation_split=0.1)

    # compute reconstruction error for other_paths
    results = []
    for p in other_paths:
        im = load_and_preprocess(p, img_size).astype('float32')/255.0
        im = im[np.newaxis,...,np.newaxis]
        recon = ae.predict(im)
        err = np.mean((recon - im)**2)
        results.append((p, err))

    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    print('Top anomaly candidates (path, reconstruction_error):')
    for r in results_sorted[:10]:
        print(r)

    return results_sorted


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised clustering on chest_xray dataset')
    parser.add_argument('--folder', type=str, default=TRAIN_DIR, help='Folder to run on (train/val/test)')
    parser.add_argument('--max-per-class', type=int, default=MAX_PER_CLASS,
                        help='Max images per class (-1 for unlimited)')
    parser.add_argument('--method', type=str, default=FEATURE_METHOD, choices=['hog', 'pixels', 'multilayer', 'efnet'],
                        help='Feature extraction method (use "efnet" to extract pretrained CNN embeddings)')
    parser.add_argument('--include-folders', type=str, default='train,val',
                        help='Comma-separated folders to use for fitting (e.g. train,val). Default prefers train then val')
    parser.add_argument('--supervised', action='store_true', help='Train a supervised classifier on embeddings (uses labels from include-folders)')
    parser.add_argument('--predict-test', action='store_true', default=True, help='Assign/predict labels for images in test/ folder')
    parser.add_argument('--save-predictions', type=str, default='test_predictions.csv', help='CSV path to save test predictions')
    parser.add_argument('--map-threshold', type=float, default=0.5, help='Pneumonia fraction threshold for mapping clusters to PNEUMONIA')
    parser.add_argument('--clusterer', type=str, default='kmeans', choices=['kmeans','gmm','agglo'], help='Which clusterer to use')
    parser.add_argument('--prob-threshold', type=float, default=0.0, help='GMM posterior probability threshold for assignment (0-1)')
    parser.add_argument('--clusters', type=int, default=N_CLUSTERS, help='Number of clusters for KMeans')
    parser.add_argument('--autoencoder', action='store_true', help='Run autoencoder anomaly detection (requires TensorFlow)')
    parser.add_argument('--no-preprocess', action='store_true', help='Disable CLAHE/denoise preprocessing')
    parser.add_argument('--normalize', action='store_true', help='Normalize resized images to 0-1 before feature extraction')
    parser.add_argument('--purity-threshold', type=float, default=0.6, help='Minimum majority fraction to map a cluster to a label (0-1)')
    parser.add_argument('--uncert-dist-mult', type=float, default=2.0, help='Multiplier for kmeans distance-based ambiguity threshold')
    parser.add_argument('--no-tsne', action='store_true', help='Skip t-SNE visualization (speeds up runs)')
    parser.add_argument('--cache-features', action='store_true', help='Cache computed features to speed repeated runs')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs for feature extraction (joblib)')
    parser.add_argument('--center-weight', type=float, default=CENTER_WEIGHT, help='Weight multiplier for center region features')
    parser.add_argument('--lung-weight', type=float, default=LUNG_WEIGHT, help='Weight multiplier for left/right lung region features')
    parser.add_argument('--peripheral-weight', type=float, default=PERIPHERAL_WEIGHT, help='Weight for upper/lower regions')
    parser.add_argument('--full-weight', type=float, default=FULL_WEIGHT, help='Weight for full-image region features')
    parser.add_argument('--cnn-model', type=str, default='efficientnet', choices=['efficientnet','mobilenet'], help='CNN model to use for embeddings (EfficientNet via TF or MobileNet via PyTorch)')
    parser.add_argument('--cnn-batch-size', type=int, default=32, help='Batch size when extracting CNN embeddings')
    args = parser.parse_args()

    # set module flags from CLI (assign into module globals to avoid local/global issues)
    globals()['PREPROCESS'] = not args.no_preprocess
    globals()['NORMALIZE'] = args.normalize
    globals()['PURITY_THRESHOLD'] = float(args.purity_threshold)
    globals()['UNCERT_DIST_MULT'] = float(args.uncert_dist_mult)
    globals()['NO_TSNE'] = bool(args.no_tsne)
    globals()['CACHE_FEATURES'] = bool(args.cache_features)
    globals()['N_JOBS'] = int(args.n_jobs)
    globals()['CENTER_WEIGHT'] = float(args.center_weight)
    globals()['LUNG_WEIGHT'] = float(args.lung_weight)
    globals()['PERIPHERAL_WEIGHT'] = float(args.peripheral_weight)
    globals()['FULL_WEIGHT'] = float(args.full_weight)
    globals()['CNN_MODEL'] = str(args.cnn_model)
    globals()['CNN_BATCH_SIZE'] = int(args.cnn_batch_size)

    print('Running unsupervised pipeline using folders:', args.include_folders)
    include_folders = [f.strip() for f in args.include_folders.split(',') if f.strip()]

    # collect images and labels from include_folders
    # Preference: when a per-class limit is set, fill from earlier folders first (e.g. train then val)
    def list_images_from_folders(folders, max_per_class=MAX_PER_CLASS):
        all_paths = []
        all_labels = []
        # discover classes present across folders
        classes = set()
        for f in folders:
            folder_path = os.path.join(BASE_DIR, f)
            if not os.path.isdir(folder_path):
                continue
            for cls in os.listdir(folder_path):
                cls_dir = os.path.join(folder_path, cls)
                if os.path.isdir(cls_dir):
                    classes.add(cls)
        classes = sorted(list(classes))

        # For each class, iterate folders in order and collect images up to max_per_class
        for cls in classes:
            count = 0
            for f in folders:
                folder_cls_dir = os.path.join(BASE_DIR, f, cls)
                if not os.path.isdir(folder_cls_dir):
                    continue
                for fname in sorted(os.listdir(folder_cls_dir)):
                    if max_per_class != -1 and count >= max_per_class:
                        break
                    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    p = os.path.join(folder_cls_dir, fname)
                    all_paths.append(p)
                    all_labels.append(cls)
                    count += 1
                if max_per_class != -1 and count >= max_per_class:
                    break
        return all_paths, all_labels

    train_paths_all, train_labels_all = list_images_from_folders(include_folders, max_per_class=args.max_per_class)

    if len(train_paths_all) == 0:
        print('No images found in include-folders:', include_folders)
        exit(1)

    print(f'Using {len(train_paths_all)} images from {include_folders} for fitting')

    # extract features/embeddings for training set
    X_train = extract_features(train_paths_all, method=args.method)

    # scale + PCA
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    pca = PCA(n_components=min(50, Xs_train.shape[1]))
    Xp_train = pca.fit_transform(Xs_train)

    # Choose supervised or unsupervised fitting
    if args.supervised:
        # Train a simple linear SVM on embeddings
        from sklearn.svm import SVC
        label_names = sorted(list(set(train_labels_all)))
        label_to_int = {lab: i for i, lab in enumerate(label_names)}
        y_train = np.array([label_to_int[l] for l in train_labels_all])
        clf = SVC(kernel='linear', probability=True)
        clf.fit(Xp_train, y_train)
        model_fitted = clf
        fitted_kind = 'supervised'
    else:
        # Unsupervised: fit clusterer on Xp_train
        if args.clusterer == 'kmeans':
            model = KMeans(n_clusters=args.clusters, random_state=42)
            clusters_train = model.fit_predict(Xp_train)
            # compute centroid-based train distance threshold for ambiguity detection
            try:
                centroids = model.cluster_centers_
                train_dists = np.linalg.norm(Xp_train - centroids[clusters_train], axis=1)
                train_dist_thresh = float(train_dists.mean() + UNCERT_DIST_MULT * train_dists.std())
            except Exception:
                centroids = None
                train_dist_thresh = None
        elif args.clusterer == 'gmm':
            model = GaussianMixture(n_components=args.clusters, reg_covar=1e-3, random_state=42)
            model.fit(Xp_train)
            probs_train = model.predict_proba(Xp_train)
            maxp = probs_train.max(axis=1)
            clusters_train = probs_train.argmax(axis=1)
            if args.prob_threshold > 0:
                clusters_train = np.where(maxp >= args.prob_threshold, clusters_train, -1)
        elif args.clusterer == 'agglo':
            model = AgglomerativeClustering(n_clusters=args.clusters)
            clusters_train = model.fit_predict(Xp_train)
        else:
            raise ValueError('Unknown clusterer')
        model_fitted = model
        fitted_kind = 'unsupervised'

    # If predicting test, prepare test features
    if args.predict_test:
        test_imgs, test_labels, test_paths = list_images(TEST_DIR, max_per_class=-1)
        if len(test_paths) == 0:
            print('No test images found in', TEST_DIR)
            test_paths = []
        else:
            X_test = extract_features(test_paths, method=args.method)
            Xs_test = scaler.transform(X_test)
            Xp_test = pca.transform(Xs_test)

    # If unsupervised, map clusters -> labels using training labels
    if fitted_kind == 'unsupervised':
        # mapping by majority-vote to detailed labels (NORMAL/PNEUMONIA)
        mapping = {}
        cluster_label_counts_train = {}
        cluster_ids = np.unique(clusters_train)
        for c in cluster_ids:
            idxs = np.where(clusters_train == c)[0]
            if len(idxs) == 0:
                continue
            assigned = [train_labels_all[i] for i in idxs]
            cnt = Counter(assigned)
            most_common_label, most_common_count = cnt.most_common(1)[0]
            purity = most_common_count / len(idxs)
            if purity >= PURITY_THRESHOLD:
                mapping[int(c)] = most_common_label
            else:
                mapping[int(c)] = 'UNKNOWN'
            cluster_label_counts_train[int(c)] = dict(cnt)
        print('Cluster -> label mapping from training set (majority vote):', mapping)

        # --- Show training-set splits and visualizations (restore previous behavior) ---
        try:
            # Print explicit split counts for cluster 0 and 1 (helps comparison)
            print('\nTraining set cluster splits (explicit clusters 0 and 1):')
            for tcid in [0, 1]:
                idxs = np.where(clusters_train == tcid)[0]
                if len(idxs) == 0:
                    print(f'Cluster {tcid}: no samples in training set')
                    continue
                mapped = mapping.get(int(tcid), 'UNKNOWN')
                counts = cluster_label_counts_train.get(int(tcid), {})
                # prepare readable counts (NORMAL/PNEUMONIA)
                normal_ct = counts.get('NORMAL', 0)
                pneu_ct = counts.get('PNEUMONIA', 0)
                print(f"Cluster {tcid} (mapped -> {mapped}): total={len(idxs)} | NORMAL={normal_ct} | PNEUMONIA={pneu_ct}")

            # PCA 2D plot of training embeddings (shows all clusters present)
            pca2 = PCA(n_components=2)
            X2 = pca2.fit_transform(Xp_train)
            plt.figure(figsize=(6,6))
            for c in np.unique(clusters_train):
                idx = clusters_train == c
                plt.scatter(X2[idx,0], X2[idx,1], label=f'cluster {int(c)}', alpha=0.6)
            plt.legend()
            plt.title('Training embeddings: PCA 2D view of clusters')
            plt.tight_layout()
            plt.show()

            # t-SNE (may be slow)
            try:
                if not NO_TSNE:
                    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
                    Xt = tsne.fit_transform(Xp_train)
                    plt.figure(figsize=(6,6))
                    for c in np.unique(clusters_train):
                        idx = clusters_train == c
                        plt.scatter(Xt[idx,0], Xt[idx,1], label=f'cluster {int(c)}', alpha=0.6)
                    plt.legend()
                    plt.title('Training embeddings: t-SNE view of clusters')
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print('t-SNE failed or is slow on your machine (training set):', e)

            # show sample images for clusters 0 and 1 from the training set
            samples_per_cluster = 6
            for c in [0, 1]:
                idxs = np.where(clusters_train == c)[0]
                if len(idxs) == 0:
                    print(f'No training images to show for cluster {c}')
                    continue
                pick = idxs[:samples_per_cluster]
                plt.figure(figsize=(12, 4))
                for i, pi in enumerate(pick):
                    img = cv2.imread(train_paths_all[pi])
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.subplot(1, samples_per_cluster, i+1)
                    plt.imshow(img)
                    title = os.path.basename(train_paths_all[pi])
                    title = f'{train_labels_all[pi]}\n{title}'
                    plt.title(title, fontsize=8)
                    plt.axis('off')
                plt.suptitle(f'Training set - Cluster {c} sample images')
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print('Could not produce training visualizations:', e)

    # Predict/assign test images
    predictions = []
    if args.predict_test and len(test_paths) > 0:
        if fitted_kind == 'supervised':
            y_prob = model_fitted.predict_proba(Xp_test)
            y_pred = model_fitted.predict(Xp_test)
            for pth, pred_idx in zip(test_paths, y_pred):
                predictions.append((pth, label_names[int(pred_idx)]))
        else:
            # unsupervised: assign clusters to test
            if args.clusterer == 'kmeans':
                test_clusters = model_fitted.predict(Xp_test)
                # mark ambiguous test points if distance to assigned centroid is large
                try:
                    if centroids is not None and train_dist_thresh is not None:
                        test_dists = np.linalg.norm(Xp_test - centroids[test_clusters], axis=1)
                        test_clusters = np.where(test_dists > train_dist_thresh, -1, test_clusters)
                except Exception:
                    pass
            elif args.clusterer == 'gmm':
                probs_test = model_fitted.predict_proba(Xp_test)
                maxp_test = probs_test.max(axis=1)
                test_clusters = probs_test.argmax(axis=1)
                if args.prob_threshold > 0:
                    test_clusters = np.where(maxp_test >= args.prob_threshold, test_clusters, -1)
            elif args.clusterer == 'agglo':
                # assign by nearest cluster centroid in PCA space
                centroids = []
                for c in cluster_ids:
                    idxs = np.where(clusters_train == c)[0]
                    if len(idxs) == 0:
                        centroids.append(np.zeros(Xp_train.shape[1]))
                    else:
                        centroids.append(Xp_train[idxs].mean(axis=0))
                centroids = np.vstack(centroids)
                dists = np.linalg.norm(Xp_test[:, None, :] - centroids[None, :, :], axis=2)
                test_clusters = dists.argmin(axis=1)
            for pth, c in zip(test_paths, test_clusters):
                mapped = mapping.get(int(c), 'UNKNOWN') if c != -1 else 'UNKNOWN'
                predictions.append((pth, mapped, int(c) if isinstance(c, (int, np.integer)) else c))

        # save predictions CSV
        outp = args.save_predictions
        with open(outp, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'predicted_label', 'predicted_cluster'])
            for row in predictions:
                if len(row) == 2:
                    writer.writerow([row[0], row[1], 'NA'])
                else:
                    writer.writerow([row[0], row[1], row[2]])
        print(f'Saved test predictions to {outp}')

        # If unsupervised clustering was used, show per-cluster split on the test set
        try:
            if fitted_kind == 'unsupervised':
                # test_clusters should be defined in the unsupervised branch above
                tc = np.array(test_clusters)
                tlabels = np.array(test_labels)
                unique_c = np.unique(tc)
                print('\nTest set split by cluster (true label counts):')
                for c in sorted(unique_c):
                    idxs = np.where(tc == c)[0]
                    if len(idxs) == 0:
                        continue
                    mapped = mapping.get(int(c), 'UNKNOWN') if c != -1 else 'UNKNOWN'
                    # count detailed labels in this cluster
                    assigned = [tlabels[i] for i in idxs]
                    cnt = Counter(assigned)
                    normal_ct = cnt.get('NORMAL', 0)
                    pneu_ct = cnt.get('PNEUMONIA', 0)
                    print(f"Cluster {int(c)} (mapped -> {mapped}): total={len(idxs)} | NORMAL={normal_ct} | PNEUMONIA={pneu_ct}")
                # Detailed test diagnostics: precision/recall/F1 and misclassified samples per cluster
                try:
                    mapped_preds_test = [mapping.get(int(c), None) if c != -1 else None for c in tc]
                    # Evaluate multi-class mapped predictions vs true detailed labels
                    valid_idx = [i for i, mp in enumerate(mapped_preds_test) if mp is not None]
                    if len(valid_idx) > 0:
                        y_true_sel = [tlabels[i] for i in valid_idx]
                        y_pred_sel = [mapped_preds_test[i] for i in valid_idx]
                        try:
                            from sklearn.metrics import classification_report, confusion_matrix
                            print('\nTest classification report (mapped predictions vs true):')
                            print(classification_report(y_true_sel, y_pred_sel, zero_division=0))
                            labels_union = sorted(list(set(y_true_sel) | set(y_pred_sel)))
                            cm_test = confusion_matrix(y_true_sel, y_pred_sel, labels=labels_union)
                            print('Confusion matrix (rows=true labels, cols=predicted labels):')
                            print('labels order:', labels_union)
                            print(cm_test)
                        except Exception as e:
                            print('Could not compute sklearn multiclass diagnostics:', e)
                    else:
                        print('No non-ambiguous test predictions to evaluate.')

                    # Save and show misclassified samples per cluster (multi-class)
                    for c in sorted(np.unique(tc)):
                        idxs = np.where(tc == c)[0]
                        if len(idxs) == 0:
                            continue
                        mis_idx = [i for i in idxs if mapped_preds_test[i] is not None and mapped_preds_test[i] != tlabels[i]]
                        if len(mis_idx) == 0:
                            continue
                        # show up to 6 misclassified examples
                        pick = mis_idx[:6]
                        plt.figure(figsize=(12, 4))
                        for i, pi in enumerate(pick):
                            img = cv2.imread(test_paths[pi])
                            if img is None:
                                continue
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            plt.subplot(1, 6, i+1)
                            plt.imshow(img)
                            gt = tlabels[pi]
                            pred = mapped_preds_test[pi] if mapped_preds_test[pi] is not None else 'UNKNOWN'
                            title = f'{gt}\n{pred}\n{os.path.basename(test_paths[pi])}'
                            plt.title(title, fontsize=8)
                            plt.axis('off')
                        plt.suptitle(f'Misclassified examples in test - Cluster {int(c)} (showing {len(pick)} of {len(mis_idx)})')
                        plt.tight_layout()
                        out_img = os.path.join(BASE_DIR, f'misclassified_cluster_{int(c)}.png')
                        try:
                            plt.savefig(out_img)
                            print(f'Saved misclassified examples for cluster {int(c)} to {out_img}')
                        except Exception:
                            pass
                        plt.show()
                except Exception as e:
                    print('Could not compute detailed test diagnostics:', e)
            else:
                print('\nNo cluster splits to show: model was trained supervised.')
        except NameError:
            # if test_clusters or mapping not present for some reason
            print('\nCould not compute test cluster splits (missing variables).')

    # If not predicting, fall back to original clustering visualization on first include folder
    if not args.predict_test:
        clusters = run_clustering_on_folder(include_folders[0], max_per_class=args.max_per_class, feature_method=args.method, n_clusters=args.clusters)

    if args.autoencoder:
        normal_folder = os.path.join(TRAIN_DIR, 'NORMAL')
        all_imgs, all_labels, all_paths = list_images(TRAIN_DIR, max_per_class=args.max_per_class)
        # exclude normal set from other_paths
        other_paths = [p for p in all_paths if '/NORMAL/' not in p.replace('\\', '/')]
        simple_autoencoder_anomaly_detection(normal_folder, other_paths, epochs=8)
