"""Retrieval index for a prepared dataset: raw feature vectors plus a FAISS index."""

import math
import os

import numpy as np

# Features above this count are clustered down, matching RVC-Project's train_index.py.
CLUSTER_THRESHOLD = 200000
CLUSTER_COUNT = 10000
ADD_BATCH = 8192


def build_index(dataset, vectors_path, index_path, log):
    """Write the raw feature vectors and, when faiss is available, a FAISS index."""
    features = np.concatenate([np.load(path) for path in dataset.feature_paths], axis=0)
    if features.shape[0] > CLUSTER_THRESHOLD:
        features = _cluster(features, log)
    vectors = np.ascontiguousarray(features, dtype=np.float32)
    np.save(vectors_path, vectors)
    log("[索引] 特征向量：%s（%s 条 × %s 维）" % (os.path.basename(vectors_path), vectors.shape[0], vectors.shape[1]))

    try:
        import faiss
    except ImportError:
        log("[索引] 未安装 faiss，跳过 %s" % os.path.basename(index_path))
        return

    n_ivf = max(1, min(int(16 * math.sqrt(vectors.shape[0])), vectors.shape[0] // 39))
    index = faiss.index_factory(vectors.shape[1], "IVF%s,Flat" % n_ivf)
    faiss.extract_index_ivf(index).nprobe = 1
    index.train(vectors)
    for start in range(0, vectors.shape[0], ADD_BATCH):
        index.add(vectors[start : start + ADD_BATCH])
    faiss.write_index(index, index_path)
    log("[索引] FAISS 索引：%s（IVF%s,Flat）" % (os.path.basename(index_path), n_ivf))


def _cluster(features, log):
    from sklearn.cluster import MiniBatchKMeans

    log("[索引] 特征 %s 条，聚类为 %s 个中心" % (features.shape[0], CLUSTER_COUNT))
    kmeans = MiniBatchKMeans(
        n_clusters=CLUSTER_COUNT,
        batch_size=256 * (os.cpu_count() or 1),
        compute_labels=False,
        init="random",
    )
    return kmeans.fit(features).cluster_centers_
