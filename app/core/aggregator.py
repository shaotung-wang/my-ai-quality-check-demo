"""
app.core.aggregator

Copied from top-level aggregator.py
"""
from collections import deque, defaultdict
from typing import Dict, Any, Tuple
import numpy as np
from app.core import config as core_config


class Aggregator:
    def __init__(self):
        # 每个 shaft 对应一个 deque 保存最近分数及触发帧的元数据
        self.buffers: Dict[Any, deque] = defaultdict(lambda: deque(maxlen=core_config.AGG_WINDOW_SIZE))

    def add(self, shaft_id, score: float, meta: dict = None):
        """添加一帧的分数与可选元数据到对应轴的 buffer"""
        entry = {'score': float(score), 'meta': meta}
        self.buffers[shaft_id].append(entry)

    def decide(self, shaft_id) -> Tuple[str, dict]:
        """对指定 shaft 做聚合判定，返回 (decision, info)

        decision: 'NG' 或 'OK'
        info: 包含用于判定的统计信息
        """
        buf = self.buffers.get(shaft_id, None)
        if not buf or len(buf) == 0:
            return 'OK', {'reason': 'no_data'}

        scores = np.array([e['score'] for e in buf], dtype=float)

        strat = core_config.AGGREGATION_STRATEGY
        if strat == 'max':
            agg_score = float(np.max(scores))
        elif strat == 'topk_mean':
            k = min(core_config.AGG_TOPK, len(scores))
            agg_score = float(np.mean(np.sort(scores)[-k:]))
        elif strat == 'quantile':
            q = core_config.AGG_QUANTILE
            agg_score = float(np.quantile(scores, q))
        else:
            agg_score = float(np.max(scores))

        # 简单决策阈值：与 CONF_THRESHOLD 比较（保守策略）
        is_ng = agg_score >= core_config.CONF_THRESHOLD

        info = {
            'agg_score': agg_score,
            'count': len(scores),
            'raw_scores': scores.tolist()
        }

        # 清理 buffer（如果需要保持状态，可改为不清理）
        self.buffers[shaft_id].clear()

        return ('NG' if is_ng else 'OK'), info


if __name__ == '__main__':
    # 简单自测
    ag = Aggregator()
    for i, s in enumerate([0.0, 0.1, 0.2, 0.05, 0.3]):
        ag.add('shaft1', s, {'frame': i})
    d, info = ag.decide('shaft1')
    print(d, info)

