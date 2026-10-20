"""Feature-index retrieval for RVC inference.

Supports the two file forms RVC-Project exports: raw ``.npy`` feature matrices
and FAISS ``.index`` files. For the FAISS form the top-k search returns squared
L2 distances, so the same 1/dist^4 weighting as the matrix path applies.
faiss is imported lazily so .npy loading works without it.
"""

import numpy as np
import torch


class IndexVectors:
    """Either 'feats' (a [N, D] torch matrix on CPU) or a FAISS index ('faiss')."""

    def __init__(self, kind, data):
        self.kind = kind
        self.data = data


_FAISS_MAGICS = (b"IwFt", b"IwFl")


def load_index(path):
    """Load a .npy feature matrix or FAISS .index file."""
    lower = path.lower()
    if lower.endswith(".npy"):
        vectors = torch.from_numpy(np.load(path)).float()
        if vectors.dim() != 2:
            raise RuntimeError("ERROR: %s is not an RVC feature matrix (expected 2-D)" % path)
        return IndexVectors("feats", vectors.contiguous().float())

    if lower.endswith(".index"):
        try:
            import faiss
        except ImportError:
            raise RuntimeError(
                "ERROR: %s is a FAISS index but faiss-cpu is not installed. "
                "Run without Index, or `pip install faiss-cpu`, or export the "
                "vectors once as .npy." % path)
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic not in _FAISS_MAGICS:
            raise RuntimeError("ERROR: %s is not a FAISS index file" % path)
        index = faiss.read_index(path)
        try:
            faiss.extract_index_ivf(index).nprobe = 1
        except Exception:
            pass
        try:
            # IVF indexes read from disk may carry an unset direct map, which
            # reconstruct_batch needs for id->vector lookup; enable Array mode.
            faiss.extract_index_ivf(index).set_direct_map_type(faiss.DirectMap.Array)
        except Exception:
            pass
        return IndexVectors("faiss", index)

    raise RuntimeError("ERROR: unsupported index file: %s (use .npy or .index)" % path)


def _retrieve_matrix(feats, vectors, k=8):
    dists = torch.cdist(feats.float(), vectors.float()).pow(2)
    k = min(k, vectors.shape[0])
    score, ix = dists.topk(k, largest=False)
    weight = torch.square(1 / score.clamp_min(1e-8))
    weight = weight / weight.sum(dim=1, keepdim=True)
    return (vectors[ix] * weight.unsqueeze(-1)).sum(dim=1), weight


def _retrieve_faiss(feats, index, k=8):
    import faiss
    ntotal = max(int(index.ntotal), 1)
    k = min(k, ntotal)
    dist, labels = index.search(feats.float().numpy(), k)
    # With nprobe=1 a list may hold fewer than k vectors; faiss fills the
    # missing slots with label -1, which reconstruct_batch rejects.
    valid = np.asarray(labels) >= 0
    flat = np.where(valid, labels, 0).reshape(-1)
    rec = index.reconstruct_batch(flat).astype("float32")
    retrieved = rec.reshape(feats.shape[0], k, -1)
    weight = torch.square(1 / torch.from_numpy(dist).clamp_min(1e-8))
    weight = weight * torch.from_numpy(valid.astype("float32"))
    weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
    vec = (torch.from_numpy(retrieved) * weight.unsqueeze(-1)).sum(dim=1)
    return vec, weight


def blend_with_index(feats, index, index_rate, k=8):
    """Blend [T, D] CPU float feats with retrieved vectors; returns [T, D]."""
    feats = feats.float()
    if index is None or index_rate <= 0:
        return feats
    if index.kind == "feats":
        retrieved, _ = _retrieve_matrix(feats, index.data, k)
    else:
        retrieved, _ = _retrieve_faiss(feats, index.data, k)
    return retrieved.to(feats.dtype) * index_rate + feats * (1 - index_rate)
