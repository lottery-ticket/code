import sys

def show():
    return '--show' in sys.argv

def save():
    return '--save' in sys.argv
