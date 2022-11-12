class Primitive:
    @classmethod
    def rect(cls):
        return Primitive()

    @classmethod
    def circle(cls):
        return Primitive()

    @classmethod
    def bezier(cls):
        return Primitive()


class SDF:
    @classmethod
    def from_primitive(cls, obj: Primitive):
        return SDF()

    def smooth_min(self, obj: Primitive, factor: float):
        return self


class Layer:
    @classmethod
    def from_sdf(cls, sdf: SDF):
        return Layer()

    def blur(self):
        return self


class Canvas:
    def __init__(self, size):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_layer(self, layer: Layer):
        return self

    def save(self):
        pass
