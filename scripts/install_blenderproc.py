import os

from visualizations.env import REPO_ROOT

if __name__ == '__main__':
    os.system(f"blenderproc pip uninstall visualizations")
    os.system(f"blenderproc pip install {REPO_ROOT}")

    with open(f"{REPO_ROOT}/requirements_blenderproc.txt") as f:
        for requirement in f.readlines():
            os.system(f"blenderproc pip install {requirement}")