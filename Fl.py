from __future__ import annotations

import math
from math import log, floor

# MACHINE SETTINGS
b = 10
t = 4
k1 = -3
k2 = 3


class Fl:
    value: float

    sinal: int
    m: float
    e: int

    def __init__(self, x):

        if isinstance(x, Fl):
            self.value = x.value
            self.sinal = x.sinal
            self.m = x.m
            self.e = x.e
            return

        # casos especiais
        if x == 0:
            self.value = 0
            self.sinal = 1
            self.m = 0
            self.e = 0
            return

        if math.isinf(x):
            self.value = float('inf')
            return

        # sinal
        self.sinal = -1 if x < 0 else 1
        x = abs(x)

        # estimativa inicial
        e = int(floor(log(x, b))) + 1
        m = x / (b ** e)

        # arredondamento
        m = round(m, t)

        # RENORMALIZAÇÃO (essencial)
        if m != 0:
            while m >= 1:
                m /= b
                e += 1

            while m < 0.1:
                m *= b
                e -= 1

        # normal
        if k1 <= e <= k2:
            pass

        # subnormal
        elif e < k1:
            e = k1
            m = x / (b ** e)
            m = round(m, t)

            # se zerar → underflow real
            if m == 0:
                self.value = 0
                self.m = 0
                self.e = 0
                return

        # overflow
        else:  # e > k2
            self.value = float('inf')
            return

        # valor final
        self.m = m
        self.e = e
        self.value = self.sinal * self.m * (b ** self.e)

    # =====================
    # operações
    # =====================

    def __add__(self, other):
        if isinstance(other, Fl):
            return Fl(self.value + other.value)
        return Fl(self.value + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return Fl(-self.value)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return Fl(other).__sub__(self)

    def __mul__(self, other):
        if isinstance(other, Fl):
            return Fl(self.value * other.value)
        return Fl(self.value * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Fl):
            return Fl(self.value / other.value)
        return Fl(self.value / other)

    def __rtruediv__(self, other):
        return Fl(other).__truediv__(self)

    # =====================
    # representação
    # =====================

    def __repr__(self):
        if self.value == float('inf') or self.value == 0:
            return str(self.value)

        sign = '-' if self.sinal == -1 else '+'
        m_str = f"{self.m:.{t}f}"

        return f"{sign}{m_str}*{b}^{self.e}"

    # =====================
    # comparações
    # =====================

    def __eq__(self, other):
        if isinstance(other, Fl):
            return self.value == other.value
        return self.value == Fl(other).value

    def __lt__(self, other):
        if isinstance(other, Fl):
            return self.value < other.value
        return self.value < Fl(other).value

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __abs__(self):
        return Fl(abs(self.value))

