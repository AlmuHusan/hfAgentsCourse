import os
from typing import Annotated
import requests
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
import json
hf_token = os.getenv("HF_TOKEN")
config = {
    "recursion_limit": 50
}
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
import pandas as pd
import base64
outputData = pd.DataFrame(
        columns=["Question", "ID","Answer"])

@tool
def readPythonFile(fileName: str) -> str:
    """Given the file name read the contents of the text file."""
    print("FILENAME"+fileName)
    with open("./data/"+fileName, 'r') as file:
        content=file.read()
        return content
@tool
def analyzeImage(imageName: str,question:str) -> str:
    """Analyze a given image based on a given question"""
    print("IMAGENAME"+imageName)
    image_data = base64.b64encode("./data/"+imageName).decode("utf-8")
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": question,
            },
            {
                "type": "image",
                "source_type": "base64",
                "data": image_data,
                "mime_type": "image/jpeg",
            },
        ],
    }
    response = llm.invoke([message])
    print(response.text())
    return response.text()
tools = load_tools(["searx-search"],
                    searx_host="http://localhost:8181/",
                    engines=["google"],
                   num_results=5)
tools.extend([readPythonFile,analyzeImage])

llm = init_chat_model(temperature=.6, model_provider='openai',
                               model="gpt-4.1-mini", streaming=False)
llm_with_tools = llm.bind_tools(tools)
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def chatbot(state: State):

    return {"messages": [llm_with_tools.invoke(state["messages"],config)]}
def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls: #,think=False
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"],config
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
graph_builder = StateGraph(State)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

graphRender=Image(graph.get_graph().draw_mermaid_png())
with open('output.png', 'wb') as f:
    f.write(graphRender.data)

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]},config=config):
        print(event.values())
        for value in event.values():
            print("Assistant:", value["messages"][-1].content.replace("*",""))
    return value
gaiaQuestions = requests.get("https://agents-course-unit4-scoring.hf.space/questions").json()
answers=[]
for q in gaiaQuestions:
    print(q["question"])
    input=q["question"]
    if(q["file_name"]!=""):
        input+="\n File Name:"+q["file_name"]
    res=stream_graph_updates(input)
    messagesT = [
        (
            "system",
            "You are a helpful assistant that provides a brief response to the user query",
        ),
        ("user", "Given the question and answer rewrite the response to only include the final answer.\n Question:"+q["question"]+"\nAnswer:"+res["messages"][-1].content.replace("*","")),
    ]
    answerFix=llm.invoke(messagesT,config)

    answers.append({"task_id": q["task_id"],
      "submitted_answer": answerFix.content
    })
    outputData.loc[len(outputData.index)] =[q["question"],q["task_id"],answerFix.content]
outputData.to_csv("./agentsTests1GPT41.csv")
finalTally=requests.post("https://agents-course-unit4-scoring.hf.space/submit", json={"username":"AlmuHusan","agent_code":"https://huggingface.co/spaces/AlmuHusan/Final_Assignment_Agent_Template/tree/main","answers":answers}).json()
print("DONE")
