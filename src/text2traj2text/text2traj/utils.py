"""Utility functions for the retail environment

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""


from typing import Any, Dict, List


def extract_intention(llm_output: Dict[str, Any]) -> List[str]:
    """extract intention from the llm output

    Args:
        llm_output (Dict[str, Any]): llm output

    Returns:
        List[str]: intention list
    """
    output = llm_output["function"]
    intentions = []
    for intention in output.intentions:
        intentions.append(intention.dict())
    return intentions
