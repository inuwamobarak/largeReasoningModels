# from fastapi import FastAPI, HTTPException
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# import torch
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer
#
# # Initialize FastAPI app
# app = FastAPI()
#
#
# # Define a request model using Pydantic for validation
# class ChatRequest(BaseModel):
#     user_input: str  # The user's input text
#     history: list  # A list to store chat history
#
#
# # Variables for model and tokenizer
# tokenizer = None
# model = None
#
#
# @app.on_event("startup")
# def load_model_and_tokenizer():
#     """
#     Load the model and tokenizer once during startup.
#     This ensures resources are initialized only once, improving efficiency.
#     """
#     global tokenizer, model
#     path = "AIDC-AI/Marco-o1"  # Path to the Marco-o1 model
#     tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
#     model = LLM(model=path, tensor_parallel_size=4)  # Parallelize model processing
#
#
# def generate_response_stream(model, text, max_new_tokens=4096):
#     """
#     Generate responses in a streaming fashion.
#     :param model: The language model to use.
#     :param text: The input prompt.
#     :param max_new_tokens: Maximum number of tokens to generate.
#     """
#     new_output = ''  # Initialize the generated text
#     sampling_params = SamplingParams(
#         max_tokens=1,  # Generate one token at a time for streaming
#         temperature=0,  # Deterministic generation
#         top_p=0.9  # Controls diversity in token selection
#     )
#     with torch.inference_mode():  # Enable efficient inference mode
#         for _ in range(max_new_tokens):  # Generate tokens up to the limit
#             outputs = model.generate(
#                 [f'{text}{new_output}'],  # Concatenate input and current output
#                 sampling_params=sampling_params,
#                 use_tqdm=False  # Disable progress bar for cleaner streaming
#             )
#             next_token = outputs[0].outputs[0].text  # Get the next token
#             new_output += next_token  # Append token to the output
#             yield next_token  # Yield the token for streaming
#
#             if new_output.endswith('</Output>'):  # Stop if the end marker is found
#                 break
#
#
# @app.post("/chat/")
# async def chat(request: ChatRequest):
#     """
#     Handle chat interactions via POST requests.
#     :param request: Contains user input and chat history.
#     :return: Streamed response or error message.
#     """
#     # Validate user input
#     if not request.user_input:
#         raise HTTPException(status_code=400, detail="Input cannot be empty.")
#
#     # Handle exit commands
#     if request.user_input.lower() in ['q', 'quit']:
#         return {"response": "Exiting chat."}
#
#     # Handle clear command to reset chat history
#     if request.user_input.lower() == 'c':
#         request.history.clear()
#         return {"response": "Clearing chat history."}
#
#     # Update history with user input
#     request.history.append({"role": "user", "content": request.user_input})
#
#     # Create the model prompt with history
#     text = tokenizer.apply_chat_template(request.history, tokenize=False, add_generation_prompt=True)
#
#     # Stream the generated response
#     response_stream = generate_response_stream(model, text)
#
#     # Return the streamed response
#     return StreamingResponse(response_stream, media_type="text/plain")


from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

app = FastAPI()


class ChatRequest(BaseModel):
    user_input: str
    history: list


tokenizer = None
model = None


@app.on_event("startup")
def load_model_and_tokenizer():
    global tokenizer, model
    path = "AIDC-AI/Marco-o1"
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = LLM(model=path, tensor_parallel_size=4)


def generate_response_stream(model, text, max_new_tokens=4096):
    new_output = ''
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        top_p=0.9
    )
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model.generate(
                [f'{text}{new_output}'],
                sampling_params=sampling_params,
                use_tqdm=False
            )
            next_token = outputs[0].outputs[0].text
            new_output += next_token
            yield next_token  # Yield each part of the response

            if new_output.endswith('</Output>'):
                break


@app.post("/chat/")
async def chat(request: ChatRequest):
    if not request.user_input:
        raise HTTPException(status_code=400, detail="Input cannot be empty.")

    if request.user_input.lower() in ['q', 'quit']:
        return {"response": "Exiting chat."}

    if request.user_input.lower() == 'c':
        request.history.clear()
        return {"response": "Clearing chat history."}

    request.history.append({"role": "user", "content": request.user_input})
    text = tokenizer.apply_chat_template(request.history, tokenize=False, add_generation_prompt=True)

    response_stream = generate_response_stream(model, text)

    # Stream the response using StreamingResponse
    return StreamingResponse(response_stream, media_type="text/plain")
