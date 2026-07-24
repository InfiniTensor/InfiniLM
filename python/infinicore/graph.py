class Graph:
    def __init__(self, underlying):
        self._underlying = underlying

    def run(self):
        self._underlying.run()
