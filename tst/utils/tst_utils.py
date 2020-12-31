from src.utils.proxy_utils import load_user_password
from src.utils.utils import dict_hierarchical


def tst_proxy_utils():
    terms = load_user_password()
    print(terms)


def tst_dict_hierarchical():
    d = dict_hierarchical()
    d.inc(["dotan", 3, (1, 1)])
    d.inc(["dotan", 3, (1, 1)])
    d.inc(["dotan", 2, (3, 1)])
    d.inc(["dotan", 2, (3, 1)])
    d.inc(["dotan", 3, (1, 2)])
    d.inc(["eitan", 3, (1, 1)])
    d.inc(["eitan", 3, (1, 1)])
    d.inc(["eitan", 2, (3, 1)])
    d.inc(["eitan", 2, (3, 1)])
    d.inc(["eitan", 3, (1, 2)])
    print(d)



if __name__ == "__main__":
    #  tst_proxy_utils()
    tst_dict_hierarchical()
