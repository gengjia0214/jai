

class DataClassDict:
    """
    Dummy class to cache the encoding
    """

    def __init__(self, names: list or str, n_classes: list or int):
        """
        Constructor. Input category order should match with the model output.
        :param names: prediction names
        :param n_classes: number of classes for the prediction
        :return: void
        """

        assert0 = isinstance(names, str) and isinstance(n_classes, int)
        if assert0:
            names = [names]
            n_classes = [n_classes]
        assert1 = isinstance(names, list) and isinstance(n_classes, list) and len(names) == len(n_classes)

        assert assert1, "names and n_classes do not match!"

        self.names = names
        self.n_classes = n_classes

    def items(self):
        return zip(self.names, self.n_classes)

