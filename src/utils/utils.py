from itertools import product


def pretty_vec(vec):
    return f"{' '.join([f'{num:+2.4f}' for num in vec])}"


class SmartCounter:

    def __init__(self):
        self.dict = dict()

    def get(self, s):
        if s not in self.dict:
            self.dict[s] = len(self.dict)
        return self.dict[s]


class dict_hierarchical:

    def __init__(self):
        self.dict = dict()

    def inc(self, keys):
        ptr = self.dict
        for i in range(len(keys)):
            if i < len(keys) - 1:
                if keys[i] not in ptr:
                    ptr[keys[i]] = dict()
                ptr = ptr[keys[i]]
            else:
                if keys[i] not in ptr:
                    ptr[keys[i]] = 0
                ptr[keys[i]] += 1

    def __str__(self, ptr=None):
        res = self._str_helper(self.dict)
        return "\n".join(res)

    def _str_helper(self, ptr):
        s = list()
        for key in ptr:
            tmp = list()
            if isinstance(ptr[key], dict):
                res = self._str_helper(ptr[key])
                tmp = tmp + res
            else:
                tmp = tmp + list(f"{ptr[key]}")
            tmp = [f"{key} -> {v}" for v in tmp]
            tmp.sort()
            s = s + tmp
        s.sort()
        return s


def product_dict(**kwargs):
    """
    Usage:
    mydict = {"x": [1, 2, 3], "y": [4, 5]}
    jobs = list(product_dict(**mydict))
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))



