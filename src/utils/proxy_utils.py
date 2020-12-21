from definitions_gnn_experiments import ROOT_DIR
from src.utils.file_utils import file_exists


def load_user_password():
    file = f"{ROOT_DIR}/login_credentials.txt"
    if not file_exists(file):
        return False
    with open(f"{ROOT_DIR}/login_credentials.txt", 'r') as file:
        data = file.read()
    terms = data.split("\n")
    terms = [term.strip() for term in terms[0:2]]
    return terms


def set_proxy():
    terms = load_user_password()
    if terms is False:
        return
    import os

    proxy = f'http://{terms[0]}:{terms[1]}@10.4.103.143:8080'

    os.environ['http_proxy'] = proxy
    os.environ['HTTP_PROXY'] = proxy
    os.environ['https_proxy'] = proxy
    os.environ['HTTPS_PROXY'] = proxy
