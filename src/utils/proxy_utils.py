from definitions_gnn_experiments import ROOT_DIR
from src.utils.file_utils import file_exists
from urllib.parse import unquote_plus


def percentage_encoding_to_regular(s):
    return unquote_plus(s)


def load_user_password(file=f"{ROOT_DIR}/login_credentials.txt"):
    if not file_exists(file):
        return False
    with open(f"{ROOT_DIR}/login_credentials.txt", 'r') as file:
        data = file.read()
    terms = data.split("\n")
    terms = [term.strip() for term in terms[0:2]]
    user, password = terms[0], terms[1]
    return user, password
