from langchain_core.tools import tool

from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langgraph.prebuilt import create_react_agent

# from tools.db import available_tools as db_tools
from tools.fastf1 import available_tools as f1_tools
from tools.strat import read_strategy_yaml


def get_agent(model):
    validator_agent = create_react_agent(
        model=model,
        tools=[*f1_tools, read_strategy_yaml, structured_validation_response],
        prompt=(
            "You are an expert in formula 1 and a race solution validator agent.\n\n"
            "INSTRUCTIONS:\n"
            "- You will receive the current strategy and a potential change to the strategy that your supervisor has considered.\n"
            "- Assess the viability of this strategy and determine if we should simulate it or not.\n"
            "- Provide a structured_validation_response with the following format:\n"
            "  Decision: [APPROVE/REJECT]\n"
            "  Description: [Explain why the decision was made, including data sources used.]\n"
            "- Limit your analysis to at most 3 queries. If the data is insufficient, make a decision based on the available information.\n"
            "- Assist ONLY with validation-related tasks. DO NOT generate any new strategies.\n"
            "- Use the data accessible to you in PostgreSQL to validate the efficacy of the proposed change.\n"
            "- Factor in multiple scenarios such as tire performance, risks of pitting early or late, and resilience to unexpected events.\n"
            "- DO NOT validate strategies that will hurt the team. We want to win.\n"
            "- After you have determined if this change is beneficial, communicate to your supervisor WHY this is beneficial and what they should consider when implementing this strategy.\n"
            "- Be definitive in your answer do not suggest any additional data, if you lack the data necessary you can reject the suggestion but NEVER tell your supervisor they should run more tests\n"
            "- Mention the sources you used such that your peers can have an easier time. We want to help them succeed too."
        ),
        name="validator_agent",
    )

    return validator_agent


@tool("structured_validation_response", return_direct=True)
def structured_validation_response(decision: str, description: str) -> str:
    """
    Provide a structured validation response.

    Args:
        decision: The decision made (e.g., APPROVE or REJECT).
        description: A detailed explanation of the decision, including data sources used.

    Returns:
        A structured response for the supervisor.
    """
    return f"Decision: {decision}\nDescription: {description}"
