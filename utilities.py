import os

def create_directory(path: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)