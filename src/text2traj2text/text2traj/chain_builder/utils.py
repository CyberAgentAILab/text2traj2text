import json
import pathlib

current_dir = pathlib.Path(__file__).parent
DATA_DIR = current_dir.parent.parent.parent.parent / "data"


def get_item_description_list():
    # load category name and item description
    item_dir = DATA_DIR / "raw_data" / "supermarket_env" / "item"
    item_description_dict = {}

    for file_path in item_dir.iterdir():
        if file_path.suffix == ".json":
            category = file_path.stem

            item_description_dict[category] = ""

            with open(file_path) as f:
                product_list = json.load(f)["products"]
            for product in product_list:
                item_description_dict[category] += f"{product}\n"

    return item_description_dict
