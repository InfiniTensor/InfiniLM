from infinilm.core.lib import _core


class Graph:
    """
    Python wrapper around a InfiniCore Graph instance.
    """

    def __init__(self, graph: _core.Graph):
        if not isinstance(graph, _core.Graph):
            raise TypeError("Expected _infinilm.Graph")
        self._graph = graph

    def run(self):
        return self._graph.run()

    def __repr__(self):
        return f"<Graph wrapper of {self._graph!r}>"
