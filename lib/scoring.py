"""工件级异常分数聚合。"""

import numpy as np


def aggregate_patch_scores(patch_scores, method="p95", top_k=3):
    """把一张工件图的多个 patch 分数聚合成一个判定分数。

    P95 比单个最大值更不容易被正常表面的孤立反光/灰尘主导，同时仍能
    保留只覆盖少量 patch 的划痕信号。
    """
    scores = np.asarray(patch_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("patch_scores 不能为空")
    if method == "max":
        return float(scores.max())
    if method == "mean":
        return float(scores.mean())
    if method == "p95":
        return float(np.percentile(scores, 95))
    if method == "topk_mean":
        k = min(max(int(top_k), 1), scores.size)
        return float(np.mean(np.partition(scores, -k)[-k:]))
    raise ValueError(f"不支持的聚合方式: {method}")
