# 1. Define libraries


# 2. Define input and output message types for for each node.



# 3. Define LLM Node


# 4. Create graph by connecting nodes



# 5. Invoke LLM function



























## 
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import dotenv
import warnings

warnings.filterwarnings("ignore")

dotenv.load_dotenv()

## Define input and output message types for for LLM function.
class MessagesState(dict):
    messages: str
    response: str


## Define LLM Node
def LLM(state: MessagesState) -> MessagesState:
    message = state["messages"]

    llm = ChatGroq(model="openai/gpt-oss-120b")

    response = llm.invoke(message)

    return {
        "messages": message,
        "response": response.content
    }

## Connnect LLM function to messages
def create_graph()->StateGraph:
    graph = StateGraph(MessagesState)

    ## Define all nodes
    graph.add_node("LLM", LLM)

    ## Define edges between nodes
    graph.set_entry_point("LLM")

    graph.add_edge("LLM", END)

    return graph

## 
if __name__ == "__main__":
    g = create_graph()
    
    app = g.compile()

    app.get_graph().draw_mermaid_png(output_file_path="graphs/graph_simple.png")

    response = app.invoke({"messages": "What is the weather in London?"})

    print(response['response'])

