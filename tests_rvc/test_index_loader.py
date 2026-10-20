import os
import tempfile

import numpy as np
import torch

def main():
    import faiss
    from rvc_inference.index_loader import IndexVectors, load_index, blend_with_index

    rng = np.random.default_rng(42)
    feats_np = rng.standard_normal((50, 8)).astype("float32")
    idx_np = rng.standard_normal((40, 8)).astype("float32")

    with tempfile.TemporaryDirectory() as tmp:
        npy_path = os.path.join(tmp, "temp_index.npy")
        faiss_path = os.path.join(tmp, "temp_index.index")

        np.save(npy_path, idx_np)
        idx = load_index(npy_path)
        assert isinstance(idx, IndexVectors) and idx.kind == "feats"

        fa = faiss.index_factory(8, "IVF2,Flat")
        fa.train(idx_np)
        fa.add(idx_np)
        faiss.write_index(fa, faiss_path)
        idx_f = load_index(faiss_path)
        assert isinstance(idx_f, IndexVectors) and idx_f.kind == "faiss"

        feats = torch.from_numpy(feats_np)
        out = blend_with_index(feats, idx, 0.75)
        assert out.shape == feats.shape and torch.isfinite(out).all()
        out_f = blend_with_index(feats, idx_f, 0.75)
        assert out_f.shape == feats.shape and torch.isfinite(out_f).all()

        # Queries contained in the index: the 1e-8 distance clamp makes the exact
        # match dominate, so retrieved == the query rows themselves.
        idx_q = IndexVectors("feats", torch.from_numpy(idx_np[:3]))
        out_q = blend_with_index(torch.from_numpy(idx_np[:3]), idx_q, 0.75)
        expected = idx_np[:3] * 0.75 + idx_np[:3] * 0.25
        assert torch.allclose(out_q, torch.from_numpy(expected), atol=1e-4)

    print("PASS: index_loader loads .npy and .index and blends")

if __name__ == "__main__":
    main()
