# 1. Define libraries


# 2. Define input and output message types for for each node.


# 3. Define LLM Node


# 4. Define Tool Nodes


# 4. Create graph by connecting nodes


# 5. Invoke LLM function











## Define libraries
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated, List
import operator # Used for combining message history
from langchain_groq import ChatGroq
import dotenv
import warnings
from pprint import pprint

warnings.filterwarnings("ignore")

dotenv.load_dotenv()

class MessagesState(TypedDict):
    """
    Represents the state of our graph.
    The messages key is a list of messages. 'operator.add' appends new messages.
    """
    messages: Annotated[List[BaseMessage], operator.add]

def add_func(a,b):
    """Adds two numbers, a and b."""
    return float(a) + float(b)

def subtract_func(a, b):
    """Subtracts number b from number a."""
    return float(a) - float(b)

def get_weather_func(city: str) -> str:
    """Returns the weather in a specified city. Always returns a dummy response for this demonstration."""
    return f"The weather in {city} is sunny and 25°C."

add_tool = Tool(
    name="add",
    func=add_func,
    description="Adds two numbers, a and b.",
    args_schema={"a": {"type": "number"}, "b": {"type": "number"}}
)

subtract_tool = Tool(
    name="subtract",
    func=subtract_func,
    description="Subtracts number b from number a.",
    args_schema={"a": {"type": "number"}, "b": {"type": "number"}}
)

get_weather_tool = Tool(
    name="get_weather",
    func=get_weather_func,
    description="Returns the weather in a specified city.",
    args_schema={"city": {"type": "string", "description": "The city name, e.g., 'London'"}}
)

# List of all tools
TOOLS = [add_tool, subtract_tool, get_weather_tool]

## Define LLM Node
def LLM(state: MessagesState) -> MessagesState:
    message = state["messages"]

    llm = ChatGroq(model="openai/gpt-oss-120b")

    llm_with_tools = llm.bind_tools(TOOLS)

    response = llm_with_tools.invoke(message)

    return {
        "messages": [response]
    }

## Connnect LLM function to messages
def create_graph()->StateGraph:
    graph = StateGraph(MessagesState)

    ## Define all nodes
    graph.add_node("LLM", LLM)
    graph.add_node("tools", ToolNode(TOOLS))

    ## Define edges between nodes
    graph.add_edge(START, "LLM")
    graph.add_conditional_edges("LLM", tools_condition)

    graph.add_edge("tools", END)

    return graph

## Invoke LLM function
if __name__ == "__main__":
    g = create_graph()
    
    app = g.compile()

    app.get_graph().draw_mermaid_png(output_file_path="graphs/graph_agent.png")

    response = app.invoke({"messages": ["What is the weather in London?"]})

    response = app.invoke({"messages": ["What is the city I asked for?"]})

    # response = app.invoke({"messages": response['messages']+["What is the city I asked for?"]})

    print(response['messages'])
    print(response['messages'][-1].content)

