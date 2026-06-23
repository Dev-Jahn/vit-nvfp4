import torch
import torch.nn.functional as F


def knn_classify(train_feats: torch.Tensor, train_labels: torch.Tensor,
                 query_feats: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Cosine k-NN majority-vote classification. Returns predicted labels for queries."""
    tn = F.normalize(train_feats.float(), dim=1)
    qn = F.normalize(query_feats.float(), dim=1)
    sims = qn @ tn.T                                  # (Q, T)
    idx = sims.topk(min(k, tn.shape[0]), dim=1).indices
    neighbor_labels = train_labels.to(idx.device)[idx]  # (Q, k)
    return torch.mode(neighbor_labels, dim=1).values


def knn_top1_accuracy(train_feats, train_labels, query_feats, query_labels, k: int = 20) -> float:
    preds = knn_classify(train_feats, train_labels, query_feats, k)
    return (preds == query_labels.to(preds.device)).float().mean().item()
