from __future__ import absolute_import

from .STCANet3D import STCANet3D 

__factory = {
        'STCANet3D': STCANet3D
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
