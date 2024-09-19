"""user intention chain builder

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

import os

from langchain.chains.llm import LLMChain
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate

from ..output_schema import IntentionList
from ..prompt import INTENTION_PROMPT
from .output_fixing_chain import OutputFixingChain


def build_user_intention_chain(llm_model_name: str, temperature: float = 0) -> LLMChain:
    """

    Args:
        llm_model_name (str): llm model name
        temperature (float, optional): _description_. Defaults to 0.

    Returns:
        LLMChain: user intention chain
    """

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

    # define parser
    parser = PydanticOutputParser(pydantic_object=IntentionList)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=fixing_model)

    # define prompt
    prompt = PromptTemplate.from_template(
        INTENTION_PROMPT,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # create chain
    chain = OutputFixingChain(
        prompt=prompt, llm=model, parser=parser, fixing_parser=fixing_parser
    )

    return chain
