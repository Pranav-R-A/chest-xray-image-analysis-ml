import os
import argparse
import csv
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import modified


def gather_all_images(base_dir, folders, max_per_class=-1):
    paths = []
    labels = []
    test_idx = []
    for f in folders:
        folder_path = os.path.join(base_dir, f)
        imgs, labs, ps = modified.list_images(folder_path, max_per_class=max_per_class)
        start = len(paths)
        paths.extend(ps)
        labels.extend(labs)
        end = len(paths)
        if f.lower() == 'test':
            test_idx = list(range(start, end))
    return paths, labels, test_idx


def cluster_and_report(paths, labels, test_idx, method='efnet', n_clusters=3, n_jobs=1, cache=True, purity_threshold=0.6):
    print(f'Extracting features using method={method} for {len(paths)} images...')
    modified.N_JOBS = n_jobs
    modified.CACHE_FEATURES = cache
    X = modified.extract_features(paths, method=method)

    print('Scaling and PCA...')
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(50, Xs.shape[1]))
    Xp = pca.fit_transform(Xs)

    print(f'Fitting KMeans (k={n_clusters}) on combined dataset...')
    k = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = k.fit_predict(Xp)

    # map clusters -> label by majority vote across all images
    mapping = {}
    for c in np.unique(clusters):
        idxs = np.where(clusters == c)[0]
        assigned = [labels[i] for i in idxs]
        cnt = Counter(assigned)
        most_common_label, most_common_count = cnt.most_common(1)[0]
        purity = most_common_count / len(idxs)
        if purity >= purity_threshold:
            mapping[int(c)] = most_common_label
        else:
            mapping[int(c)] = 'UNKNOWN'

    # Prepare test predictions
    y_true = [labels[i] for i in test_idx]
    y_pred = [mapping.get(int(clusters[i]), 'UNKNOWN') for i in test_idx]

    print('\nClassification report (test images only):')
    print(classification_report(y_true, y_pred, zero_division=0))
    try:
        labels_union = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels_union)
        print('Confusion matrix (rows=true, cols=predicted):')
        print('labels order:', labels_union)
        print(cm)
    except Exception:
        pass

    # Save test predictions CSV
    outp = os.path.join(modified.BASE_DIR, 'final_test_predictions.csv')
    with open(outp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_label', 'predicted_label', 'predicted_cluster'])
        for i in test_idx:
            writer.writerow([paths[i], labels[i], mapping.get(int(clusters[i]), 'UNKNOWN'), int(clusters[i])])
    print(f'Saved test predictions to {outp}')


def main():
    parser = argparse.ArgumentParser(description='Run KMeans on train/val/test and report on test only')
    parser.add_argument('--folders', type=str, default='train,val,test', help='Comma-separated folders to include (default train,val,test)')
    parser.add_argument('--method', type=str, default='efnet', choices=['hog', 'pixels', 'multilayer', 'efnet'], help='Feature extraction method')
    parser.add_argument('--clusters', type=int, default=3, help='Number of KMeans clusters')
    parser.add_argument('--max-per-class', type=int, default=-1, help='Max images per class per folder (-1 for all)')
    parser.add_argument('--n-jobs', type=int, default=4, help='Parallel jobs for feature extraction')
    parser.add_argument('--no-cache', action='store_true', help='Disable feature caching')
    parser.add_argument('--purity-threshold', type=float, default=0.6, help='Purity threshold for cluster->label mapping')
    args = parser.parse_args()

    folders = [f.strip() for f in args.folders.split(',') if f.strip()]
    paths, labels, test_idx = gather_all_images(modified.BASE_DIR, folders, max_per_class=args.max_per_class)
    if len(paths) == 0:
        print('No images found in folders:', folders)
        return

    cluster_and_report(paths, labels, test_idx, method=args.method, n_clusters=args.clusters, n_jobs=args.n_jobs, cache=(not args.no_cache), purity_threshold=args.purity_threshold)


if __name__ == '__main__':
    main()
