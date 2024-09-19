"""Output schema for the retail environment

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import json
import pathlib
from typing import Dict, List, Literal, Sequence, Tuple

from langchain.pydantic_v1 import BaseModel, Field, create_model

current_dir = pathlib.Path(__file__).parent
DATA_DIR = current_dir.parent.parent.parent / "data"

CATEGORY_LITERAL = Literal[
    "alcohol",
    "dairy",
    "drink",
    "fruit",
    "household goods",
    "meat",
    "seafood",
    "seasoning",
    "snack",
    "vegetable",
]

## Intention
class Intention(BaseModel):
    intention: str = Field(..., description="The intention of the customer")
    num_item_to_buy: int = Field(
        ...,
        description="number of products the customer plan to purchase. It is carefully calculated based on various factors such as the customer's purchasing intentions, personality traits, and family composition to ensure the output aligns with the customer's actual needs and preferences.",
    )
    purchase_consideration: Literal["1", "2", "3", "4", "5"] = Field(
        ...,
        description="Degree of customer purchase consideration. It should be carefully calculated based on a variety of factors, including the customer's intention and personality. Higher values indicate a tendency to spend more time on a purchase and to compare a variety of products, while lower values indicate a tendency to spend less time on a purchase and to decide in advance which product to purchase.",
    )


class IntentionList(BaseModel):
    intentions: Sequence[Intention] = Field(
        ..., description="The list of intentions of diverse customers"
    )


## Paraphrased Intention
class ParaphrasedIntention(BaseModel):
    paraphrased_intention: str = Field(
        ...,
        description="The paraphrased text of the original text includes all information from the original text and is almost the same length as the original text.",
    )


class ParaphrasedIntentionList(BaseModel):
    paraphrased_intentions_list: Sequence[ParaphrasedIntention] = Field(
        ..., description="The list of diverse paraphrase of original text"
    )


## Preferred Item
def create_item_literal(category_file: pathlib.Path) -> Sequence:
    """create item literal for the category

    Args:
        category_file (pathlib.Path): path to the category file

    Returns:
        Sequence: sequence of item literal
    """
    with open(category_file) as f:
        all_items = json.load(f)["products"]

    item_name_list = [product["name"] for product in all_items]

    item_literal = Literal[tuple(item_name_list)]
    item_literal_list = Sequence[item_literal]
    return item_literal_list


def create_item_list_schema(literal_name: str, item_literal_list: Literal) -> BaseModel:
    """create item list schema for the category

    Args:
        literal_name (str): name of the literal
        item_literal_list (Literal): item literal list

    Returns:
        BaseModel: item list schema
    """
    annotations = {}
    annotations["inclined_to_purchase"] = (
        item_literal_list,
        Field(..., description="item list that the customer is inclined to purchase"),
    )
    annotations["show_interest"] = (
        item_literal_list,
        Field(..., description="item list that the customer is interested in"),
    )
    # Creating the model with the given name and fields
    base_model = create_model(literal_name, **annotations)

    return base_model


def create_action_plan_schema(category_list: List[str]) -> BaseModel:
    """create action plan schema for the category

    Args:
        category_list (List[str]): list of category

    Returns:
        BaseModel: action plan schema
    """
    # initialize category list literal
    category_literal = Literal[tuple(category_list)]

    # create shopping plan base model for LLM output parse
    class CategoryShoppingPlan(BaseModel):
        category: category_literal = Field(
            ...,
            description="category name that the customer is inclined to purchase or show interest in",
        )
        num_purchase_item: int = Field(..., description="number of items to purchase")

    class ShoppingPlan(BaseModel):
        shopping_plan: Sequence[CategoryShoppingPlan] = Field(
            ..., description="shopping plan that the customer is inclined to purchase"
        )

    return ShoppingPlan


def create_output_schema() -> Tuple[BaseModel, Dict[str, BaseModel]]:
    """create output schema for the retail environment

    Returns:
        Tuple[BaseModel, Dict[str, BaseModel]]: action plan schema and item list schema
    """
    # create item base model for LLM output parse
    category_list = []
    category_item_list_base_model = {}
    item_dir = DATA_DIR / "raw_data" / "supermarket_env" / "item"

    for filename in item_dir.iterdir():
        if filename.suffix == ".json":
            category = filename.stem
            category_list.append(category)

            item_list_literal = create_item_literal(filename)
            category_item_list_base_model[category] = create_item_list_schema(
                f"ItemList{category}", item_list_literal
            )

    action_plan_base_model = create_action_plan_schema(category_list)

    return action_plan_base_model, category_item_list_base_model
