# %%

import yaml

def open_file(path="test_file.yaml"):
    with open(path, "r") as file:
        data = yaml.safe_load(file)
        print(data)

if __name__ == "__main__":
    open_file()
# %%
