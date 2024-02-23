"""
Chat with File. Load a file directly into context.
"""

import gradio as gr
import textwrap
import torch
from transformers import LogitsProcessor

from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)
from extensions.chatwithfile.rag.ingest_file_class import Ingest_File
params = {
    "display_name": "Example Extension",
    "is_tab": False,
    "filecontent": []
}

class MyLogits(LogitsProcessor):
    """
    Manipulates the probabilities for the next token before it gets sampled.
    Used in the logits_processor_modifier function below.
    """
    def __init__(self):
        pass

    def __call__(self, input_ids, scores):
        # probs = torch.softmax(scores, dim=-1, dtype=torch.float)
        # probs[0] /= probs[0].sum()
        # scores = torch.log(probs / (1 - probs))
        return scores

def history_modifier(history):
    """
    Modifies the chat history.
    Only used in chat mode.
    """
    return history

def state_modifier(state):
    """
    Modifies the state variable, which is a dictionary containing the input
    values in the UI like sliders and checkboxes.
    """
    return state

def chat_input_modifier(text, visible_text, state):
    """
    Modifies the user input string in chat mode (visible_text).
    You can also modify the internal representation of the user
    input (text) to change how it will appear in the prompt.
    """
    return text, visible_text

def input_modifier(string, state, is_chat=False):
    """
    In default/notebook modes, modifies the whole prompt.

    In chat mode, it is the same as chat_input_modifier but only applied
    to "text", here called "string", and not to "visible_text".
    """
    string = "[File Contents:" + params['filecontent'] + " ]" + string
    print(string)
    return string

def bot_prefix_modifier(string, state):
    """
    Modifies the prefix for the next bot reply in chat mode.
    By default, the prefix will be something like "Bot Name:".
    """
    return string

def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    """
    Modifies the input ids and embeds.
    Used by the multimodal extension to put image embeddings in the prompt.
    Only used by loaders that use the transformers library for sampling.
    """
    return prompt, input_ids, input_embeds

def logits_processor_modifier(processor_list, input_ids):
    """
    Adds logits processors to the list, allowing you to access and modify
    the next token probabilities.
    Only used by loaders that use the transformers library for sampling.
    """
    processor_list.append(MyLogits())
    return processor_list

def output_modifier(string, state, is_chat=False):
    """
    Modifies the LLM output before it gets presented.

    In chat mode, the modified version goes into history['visible'],
    and the original version goes into history['internal'].
    """
    return string

def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """
    result = chat.generate_chat_prompt(user_input, state, **kwargs)
    return result

def custom_css():
    """
    Returns a CSS string that gets appended to the CSS for the webui.
    """
    return ''

def custom_js():
    """
    Returns a javascript string that gets appended to the javascript
    for the webui.
    """
    return ''

def setup():
    """
    Gets executed only once, when the extension is imported.
    """
    pass

def rag_upload_file(fileobj):
    file_path = str(fileobj.name)
    print("********FILEPATH*********")
    print(file_path)
    
    load_file = Ingest_File(file_path)
    file_content = load_file.loadfile()
        
        
    
    print("***********RAG FILE ADDED TO CONTEXT*************")
    print(file_content)

    print("**************************************")
    params['filecontent'] = f"[FILE_CONTENT={file_path}]\n{file_content}"
    pass


def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.

    To learn about gradio components, check out the docs:
    https://gradio.app/docs/
    """
    with gr.Blocks() as memoir_ui:
        with gr.Accordion("Chat with File"):
            with gr.Row():
                gr.Markdown(textwrap.dedent("""
            - If you find this extension useful, <a href="https://www.buymeacoffee.com/brucepro">Buy Me a Coffee:Brucepro</a> or <a href="https://ko-fi.com/brucepro">Support me on Ko-fi</a>
            - For feedback or support, please raise an issue on https://github.com/brucepro/chatwithfile
            """))
            with gr.Row():
                gr.Markdown(textwrap.dedent("""
                - The RAG system uses the langchain loaders. It sticks the file contents into the input.
                - 
                - The file box below will load the file directly to the context on next input.
                """))
                

            with gr.Row():
                rag_upload_box = gr.Interface(fn=rag_upload_file,inputs=["file",],outputs="text")
                
    pass
