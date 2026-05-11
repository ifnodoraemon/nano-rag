from __future__ import annotations

from pathlib import Path

from app.retrieval.graph_index import GraphIndex


class GraphExpander:
    def __init__(self, parsed_dir: Path) -> None:
        self.index = GraphIndex(parsed_dir)

    def expand(
        self,
        contexts: list[dict[str, object]],
        *,
        kb_id: str,
        max_neighbors: int = 8,
    ) -> list[dict[str, object]]:
        seed_node_ids = {
            str(context.get("node_id") or context.get("chunk_id"))
            for context in contexts
            if context.get("node_id") or context.get("chunk_id")
        }
        if not seed_node_ids:
            return []
        view = self.index.load(kb_id)
        expanded: list[dict[str, object]] = []
        seen = set(seed_node_ids)
        for node_id in seed_node_ids:
            for neighbor_id, relation in self.index.neighbor_node_ids(view, node_id):
                if neighbor_id in seen:
                    continue
                context = self.index.node_context(view, neighbor_id, relation)
                if context is None:
                    continue
                seen.add(neighbor_id)
                expanded.append(context)
                if len(expanded) >= max_neighbors:
                    return expanded
        return expanded
