## Define libraries
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import dotenv
import warnings

warnings.filterwarnings("ignore")

dotenv.load_dotenv()

## Define input and output message types for each node.
class MessagesState(dict):
    messages: str
    response: str


## Define LLM node
def LLM(state: MessagesState) -> MessagesState:
    message = state["messages"]

    llm = ChatGroq(model="openai/gpt-oss-120b")

    response = llm.invoke(message)

    return {
        "messages": message,
        "response": response
    }

def create_graph()->StateGraph:
    graph = StateGraph(MessagesState)

    ## Define all nodes
    graph.add_node("LLM", LLM)

    ## Define edges between nodes
    graph.set_entry_point("LLM")

    graph.add_edge("LLM", END)

    return graph

if __name__ == "__main__":
    g = create_graph()
    
    app = g.compile()

    app.get_graph().draw_mermaid_png(output_file_path="graph.png")

    response = app.invoke({"messages": "What is the capital of France?"})

    print(response['response'].content)

