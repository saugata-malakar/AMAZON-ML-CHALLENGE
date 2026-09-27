"""
Token and Inverted Index Blocking:
- Exact postal code / PIN code blocks
- Rare token / inverted index blocks
"""

from collections import defaultdict
from typing import Dict, List, Set


class TokenInvertedBlocker:
    """
    Builds an inverted index over discrete blocking keys (e.g. PIN codes or rare tokens).
    """
    def __init__(self):
        self.index: Dict[str, List[str]] = defaultdict(list)

    def add_record(self, record_id: str, keys: List[str]):
        for k in keys:
            if k:
                self.index[k].append(record_id)

    def get_candidates(self, keys: List[str], max_block_size: int = 500) -> Set[str]:
        candidates = set()
        for k in keys:
            if not k:
                continue
            matches = self.index.get(k, [])
            # Avoid huge blocks that defeat the purpose of blocking
            if 0 < len(matches) <= max_block_size:
                candidates.update(matches)
        return candidates
