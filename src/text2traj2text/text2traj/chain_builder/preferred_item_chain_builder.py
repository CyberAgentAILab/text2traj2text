"""preferred item chain builder

preferred item is calculated by two steps:
Step:
    (1) Generate shopping plan based on the user's intention and category list
    (2) Generate preferred item list based on the user's intention, shopping plan, and item list

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import os
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel
from pydantic import Extra

from ..output_schema import create_output_schema
from ..prompt import PREFERRED_ITEM_PROMPT, SHOPPING_PLAN_PROMPT
from .output_fixing_chain import OutputFixingChain
from .utils import get_item_description_list


class PreferredItemChain(Chain):
    """
    Chain for preferred item list generation.

    Step:
        1) Generate shopping plan based on the user's intention and category list
        2) Generate preferred item list based on the user's intention, shopping plan, and item list

    Returns:
        Dict[str, Dict]: preferred item list and shopping plan
    """

    shopping_plan_chain: OutputFixingChain
    preferred_item_chain: Dict[str, OutputFixingChain]
    category_list: List[str]
    item_discription_dict: Dict[str, str]
    output_key: str = "output"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the shopping_plan_prompt expects.

        :meta private:
        """
        return ["num_item", "intention", "purchase_consideration"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Dict]:
        """
        function for preferred item list generation

        preferred item is calculated by two steps:
            1) Generate action plan based on the user's intention and category list by shopping_plan_chain
            2) Generate preferred item list based on the user's intention, action plan, and item list by preferred_item_chain

        Args:
            inputs (Dict[str, Any]): input for preferred item list generation.
            run_manager (Optional[CallbackManagerForChainRun], optional): manager. Defaults to None.

        Returns:
            Dict[str, Dict]: preferred item list and action plan
        """

        # generate action plan
        shopping_planner_input = {
            "num_item": inputs["num_item"],
            "intention": inputs["intention"],
            "category_list": self.category_list,
        }
        planner_response = self.shopping_plan_chain.invoke(shopping_planner_input)

        # generate preferred item list
        incline_to_purchase = []
        show_interest = []
        for plan in planner_response["function"].shopping_plan:
            if plan.num_purchase_item > 0:
                item_list_output = self.preferred_item_chain[plan.category].invoke(
                    {
                        "category": plan.category,
                        "num_purchase_items": plan.num_purchase_item,
                        "intention": inputs["intention"],
                        "purchase_consideration": inputs["purchase_consideration"],
                        "item_description": self.item_discription_dict[plan.category],
                    },
                )
                incline_to_purchase.extend(
                    item_list_output["function"].inclined_to_purchase
                )
                show_interest.extend(item_list_output["function"].show_interest)

        preffered_item = {
            "incline_to_purchase": incline_to_purchase,
            "show_interest": show_interest,
        }

        output = {
            "preffer_item": preffered_item,
            "shopping_plan": planner_response["function"].shopping_plan,
        }

        return {self.output_key: output}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError

    @property
    def _chain_type(self) -> str:
        return "preferred_item_chain"


def _build_intention_to_shopping_plan_chain(
    model: AzureChatOpenAI,
    fixing_model: AzureChatOpenAI,
    action_plan_base_model: BaseModel,
    verbose: bool = True,
) -> LLMChain:
    """build function for shopping plan chain

    Args:
        model (AzureChatOpenAI): llm model for shopping plan generation
        fixing_model (AzureChatOpenAI): llm model for output fixing
        category_list (List[str]): category list that the super market has
        verbose (bool, optional): verbose. Defaults to True.

    Returns:
        LLMChain: shopping plan chain
    """

    parser = PydanticOutputParser(pydantic_object=action_plan_base_model)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=fixing_model)

    prompt = PromptTemplate.from_template(
        SHOPPING_PLAN_PROMPT,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = OutputFixingChain(
        prompt=prompt, llm=model, parser=parser, fixing_parser=fixing_parser
    )
    return chain


def _build_shopping_plan_to_preferred_item_list_chain(
    model: AzureChatOpenAI,
    fixing_model: AzureChatOpenAI,
    preffered_item_base_model_dict: Dict[str, BaseModel],
    verbose: bool = True,
) -> Dict[str, LLMChain]:
    """build function for preferred item list chain

    Args:
        model (AzureChatOpenAI): llm model for preferred item list generation
        fixing_model (AzureChatOpenAI): llm model for output fixing
        item_dir (str): directory path of item list
        verbose (bool, optional): verbose. Defaults to True.

    Returns:
        Dict[str, LLMChain]: preferred item list chain for each category
    """

    # create chain
    chain_dict = {}
    for category_name, base_model in preffered_item_base_model_dict.items():
        parser = PydanticOutputParser(pydantic_object=base_model)
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=fixing_model)
        prompt = PromptTemplate.from_template(
            PREFERRED_ITEM_PROMPT,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain_dict[category_name] = OutputFixingChain(
            prompt=prompt, llm=model, parser=parser, fixing_parser=fixing_parser
        )
    return chain_dict


def build_preferred_item_chain(
    llm_model_name: str, temperature: float = 0, verbose: bool = True
) -> PreferredItemChain:

    # initialized llm model
    model = AzureChatOpenAI(
        azure_deployment=llm_model_name,
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        temperature=temperature,
    )

    # output fixing llm model
    fixing_model = AzureChatOpenAI(
        azure_deployment=llm_model_name,
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        temperature=temperature,
    )

    action_plan_base_model, preffered_item_base_model_dict = create_output_schema()

    intention_to_shopping_plan_chain = _build_intention_to_shopping_plan_chain(
        model, fixing_model, action_plan_base_model, verbose
    )

    shopping_plan_to_preferred_item_list_chain = (
        _build_shopping_plan_to_preferred_item_list_chain(
            model, fixing_model, preffered_item_base_model_dict, verbose
        )
    )

    category_list = list(preffered_item_base_model_dict.keys())
    item_description_dict = get_item_description_list()

    chain = PreferredItemChain(
        shopping_plan_chain=intention_to_shopping_plan_chain,
        preferred_item_chain=shopping_plan_to_preferred_item_list_chain,
        category_list=category_list,
        item_discription_dict=item_description_dict,
    )

    return chain
