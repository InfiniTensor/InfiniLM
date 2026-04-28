from infinicore.lib import _infinicore


class Graph:
    """
    Python wrapper around a InfiniCore Graph instance.
    """

    def __init__(self, graph: _infinicore.Graph):
        if not isinstance(graph, _infinicore.Graph):
            raise TypeError("Expected _infinicore.Graph")
        self._graph = graph

    def run(self):
        return self._graph.run()

    def __repr__(self):
        return f"<Graph wrapper of {self._graph!r}>"
