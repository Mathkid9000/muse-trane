from helpers import STORAGE_DIR
from pathlib import Path
import shutil

if __name__ == "__main__":
    origin = Path("./omnibook/")
    target = STORAGE_DIR / "data/omnibook/"
    target.mkdir(parents=True, exist_ok=True)

    print(f'Copying to {str(target)}')
    for file in origin.iterdir():
        print(f'Copying {file.name}')
        shutil.copy2(file, target)
