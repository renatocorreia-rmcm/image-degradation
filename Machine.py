

class Machine:

    modes = ['round', 'truncate']

    def __init__(
            self,
            base: int,
            mantissa: int,
            k1: int, k2: int,
            mode: str
    ):
        self.base = base
        self.mantissa = mantissa
        self.k1 = k1
        self.k2 = k2
        assert mode in self.modes
        self.mode = mode


    def fl(self, x: float):
