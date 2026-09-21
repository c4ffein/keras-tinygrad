class TinygradLayer:
    pass


# `BackendLayer`: what keras' pluggable_backend branch reads from
# `keras_<name>.src.layer`. `Layer`: the older plugin-PoC name.
BackendLayer = TinygradLayer
Layer = TinygradLayer
