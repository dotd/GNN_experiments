from src.utils.proxy_utils import load_user_password


def tst_proxy_utils():
    terms = load_user_password()
    print(terms)


if __name__ == "__main__":
    tst_proxy_utils()
