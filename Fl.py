from __future__ import annotations # allow to reference type class inside herself
from math import log, floor

# MACHINE SETTINGS
b=2
t=3
k1=-5
k2=5


class Fl:

    value: float

    def __init__(self, x):

        # absolute zero
        if x == 0:
            self.value = 0
            return

        # other cases

        # scientific notation
        self.sinal = -1 if x < 0 else 1
        x = abs(x)
        self.e = int(floor(log(x, b)))
        self.m = x / (b ** self.e)
        self.m = round(self.m, t - 1)  # todo: allow truncation
        if self.m >= b:
            self.m /= b
            self.e += 1

        # infinite
        if self.e > k2:
            self.value = float('inf')
            return
        # zero
        if self.e < k1:
            self.value = 0
            return

        # ordinary representable
        self.value = self.sinal * self.m * (b ** self.e)
        return




    def __add__(self, other: Fl):
        return Fl(self.value + other.value)

    def __sub__(self, other):
        return Fl(self.value - other.value)

    def __mul__(self, other):
        return Fl(self.value * other.value)

    def __truediv__(self, other):
        return Fl(self.value / other.value)

    def __repr__(self):
        return f"{self.m}*{b}^{self.e}"



a = Fl(3.12345)

print(a)
