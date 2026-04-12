import json
import os

path = "C:\\Users\\thien\\Desktop\\Work stuff\\TensorDecomFine-tune\\dreambooth-main\\dataset\\prompts_and_classes.json"
with open(path, "r") as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))