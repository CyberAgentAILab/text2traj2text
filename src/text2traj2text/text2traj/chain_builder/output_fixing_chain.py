"""
Definition of OutputFixingChain class.
OutputFixingChain is a chain for generating text and fixing output format of language model which is based on the langchain.output_parsers.OutputFixingParser class.

OutputFixingChain is working as following steps:
Steps:
    (1) generate original output from language model
    (2) parse original output to check whether it follows the output format
    (3) if original output does not follow the output format, fix the output format using language model


Author: Hikaru Asano
Affiliation: The University of Tokyo / CyberAgent AI Lab
"""

from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import OutputParserException
from langchain.schema.language_model import BaseLanguageModel
from pydantic import Extra


class OutputFixingChain(Chain):
    prompt: BasePromptTemplate
    llm: BaseLanguageModel
    parser: PydanticOutputParser
    fixing_parser: OutputFixingParser
    output_key: str = "function"  #: :meta private:
    num_retries: int = 3
    retry_temperature: float = 0.7

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the shopping_plan_prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

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

        model_input = self.prompt.format(**inputs)

        for attempt in range(self.num_retries):
            try:
                model_output = self.llm.invoke(
                    model_input, temperature=self.retry_temperature
                )
                break
            except ValueError as e:
                if attempt == self.num_retries - 1:
                    raise
                else:
                    continue

        # parse output
        try:
            output = self.parser.parse(model_output.content)
        except OutputParserException as e:
            # if failed to parse, use llm to fix original llm-output to follow the output format
            llm_output = e.llm_output
            output = self.fixing_parser.parse(llm_output)

        return {self.output_key: output}
