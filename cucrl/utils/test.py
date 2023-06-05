from abc import abstractmethod, ABC

if __name__ == "__main__":

    class A(ABC):
        def __init__(self, a, *args, **kwargs):
            self.a = a

        @abstractmethod
        def next(self, x):
            pass

    class B:
        def __init__(self, b):
            self.b = b

    class C(B, A):
        def __init__(self, a, b, c):
            B.__init__(self, b)
            A.__init__(self, a)
            self.c = c

        def next(self, x):
            return x + 1

    test = C(c=4, b="hello", a=5)
