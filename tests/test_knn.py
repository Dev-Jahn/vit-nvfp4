import torch

from vit_nvfp4.eval.knn import knn_classify, knn_top1_accuracy


def _three_clusters(n=30, d=16, spread=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(3, d)  # 3 far-apart cluster centers
    feats, labels = [], []
    for c in range(3):
        feats.append(centers[c] + spread * torch.randn(n, d, generator=g))
        labels += [c] * n
    return torch.cat(feats), torch.tensor(labels)


def test_knn_separable_clusters_perfect():
    tf, tl = _three_clusters(seed=1)
    qf, ql = _three_clusters(seed=2)  # same clusters, different noise
    acc = knn_top1_accuracy(tf, tl, qf, ql, k=5)
    assert acc == 1.0, acc


def test_knn_classify_shapes_and_labels():
    tf, tl = _three_clusters(seed=1)
    qf, _ = _three_clusters(seed=3)
    preds = knn_classify(tf, tl, qf, k=7)
    assert preds.shape == (qf.shape[0],)
    assert set(preds.tolist()) <= {0, 1, 2}
