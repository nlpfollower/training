import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def dump_json(obj, fp, **kwargs):
    kwargs.setdefault('cls', NumpyEncoder)
    json.dump(obj, fp, **kwargs)


def dumps_json(obj, **kwargs):
    kwargs.setdefault('cls', NumpyEncoder)
    return json.dumps(obj, **kwargs)