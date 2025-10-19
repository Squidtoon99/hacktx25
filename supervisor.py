from typing import Annotated
from models import model
from context import Context

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import convert_to_messages

from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command

from langgraph.types import Send
from langgraph.prebuilt import create_react_agent

from langchain_groq import ChatGroq

from agents import validator, strategy_generator
from agents.report_generator import get_justification_agent


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=[Send(agent_name, agent_input)],
            update={**state, "messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,
        )

    return handoff_tool


assign_to_validator_agent_with_description = create_task_description_handoff_tool(
    agent_name="validator_agent",
    description="Assign task to a validator agent.",
)

assign_to_analysis_agent_with_description = create_task_description_handoff_tool(
    agent_name="analysis_agent",
    description="Assign task to the analysis agent.",
)

assign_to_implementation_agent_with_description = create_task_description_handoff_tool(
    agent_name="implementation_agent",
    description="Assign task to the implementation agent.",
)

supervisor_agent_with_description = create_react_agent(
    model=model,
    tools=[
        assign_to_validator_agent_with_description,
        assign_to_analysis_agent_with_description,
        assign_to_implementation_agent_with_description,
    ],
    prompt=(
        "You are a formula 1 race strategy supervisor. Your job is to think of new changes to the strategy when events happen on the track and determine if we should use them.:\n"
        "You will be given the current race strategy and the event that has just occured on the track. You must do the following:\n"
        "1. Brainstorm a list of potential changes to the strategy that allow us to take advantage of the unexpected event\n"
        "2. For each of the potential changes you should employ your team to validate if the change will be beneficial to the team\n"
        "3. For all valid changes, you want to generate a new strategy plan using the change and provide information to back it up\n"
        "4. Lastly communicate a list of changes and the change ids that your team has created for you back to your system so they can deploy the new strategy to the team\n\n"
        "Thankfully, you are not alone. You have specialized agents to help you in this task. You have 2 Agents available to you:"
        "- a validator agent. Assign strategy change validation tasks to this assistant. Only give this assistant one strategy change at a time. Never validate a strategy on your own, always use this agent to validate potential strategy changes\n"
        "- a strategy generator agent. Once a change has been approved by your trusted validator, you can ask your strategy generator to generate a new spec using their speciailized generators. They will also generate a unique report and convey to you the ids that you can send to your team\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not generate more than **4** different strategy change(s) since we are just testing\n"
        "Do not do any work yourself. Only come up with ideas and coordinate the execution of these ideas between your employees"
    ),
    name="supervisor",
)

supervisor_with_description = (
    StateGraph(MessagesState, context_schema=Context)
    .add_node(
        supervisor_agent_with_description,
        # destinations=("validator_agent", "analysis_agent", "implementation_agent", END),
    )
    .add_node(validator.get_agent(model))
    .add_node(strategy_generator.get_analysis_agent(model))
    .add_node(get_justification_agent(model))
    .add_node(strategy_generator.get_implementation_agent(model))
    .add_edge(START, "supervisor")
    .add_edge("validator_agent", "analysis_agent")
    # .add_edge("analysis_agent", "implementation_agent")
    .add_edge("analysis_agent", "justification_agent")
    .add_edge("justification_agent", "implementation_agent")
    .add_edge("implementation_agent", "supervisor")
    .add_edge("supervisor", END)
    .compile()
)

with open("strategy-template.yaml", "r") as fp:
    strategy = fp.readlines()

chunk = {"supervisor": {"messages": []}}
for chunk in supervisor_with_description.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": f"The current strategy is {strategy}\nNEW EVENT: Rain is estimated on the track.\nWhat is our new strategy?",
            }
        ]
    },  # type: ignore
    subgraphs=True,
    context={
        "strategy": strategy,
        "event": "Wet Tires",
        "event_description": "There is rain forecasted in the next 5 minutes with a 80% chance of precipitation.",
        "time": "2025-04-25 01:12:50.313381 +00:00",
    },  # pyright: ignore[reportArgumentType]
):
    pretty_print_messages(chunk, last_message=True)

print("\n\nFINAL MESSAGES:\n")
for cd in chunk:
    if not cd or "supervisor" not in cd:
        continue
    if final_message_history := cd["supervisor"]["messages"]:
        for message in final_message_history:
            message.pretty_print()
