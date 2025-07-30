import gradio as gr
import pandas as pd
import torch
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import io

# --- Model and Tokenizer Configuration ---
# We are defining the model path directly here for simplicity.
# This model is powerful but requires significant resources (GPU recommended).
CHECKPOINT_PATH = "Llama-3.1-8B"

def _get_args():
    parser = ArgumentParser(description="Qwen2.5-Instruct web chat demo.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=CHECKPOINT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        default=True,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()
    return args
  
def load_model_tokenizer():
    """Loads the pre-trained model and tokenizer."""
    print("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_PATH,
        resume_download=True,
    )

    # Use GPU if available, otherwise fall back to CPU.
    #device_map = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    ).eval()
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

# --- Core Chat Logic ---
def chat_stream(model, tokenizer, query, history):
    """Generates a response from the model in a streaming fashion."""
    # Apply the chat template to format the conversation history and new query
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": query})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Setup the streamer for text generation
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    
    generation_kwargs = {
        **model_inputs,
        "streamer": streamer,
        "max_new_tokens": 2048,
        "do_sample": True,
        "top_p": 0.95,
        "top_k": 50,
        "temperature": 0.7,
    }
    
    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield new text as it becomes available
    for new_text in streamer:
        yield new_text

# --- Main Application UI and Logic ---
def build_chatbot_ui(model, tokenizer):
    """Builds the Gradio web interface for the chatbot."""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".control-height { height: 500px; overflow: auto; }"
    ) as demo:
        # State management
        df_state = gr.State(None) # To hold the pandas DataFrame
        task_history = gr.State([]) # To hold the conversation history

        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1>📊 dennislee CSV Analysis Chatbot</h1>
                <p>Upload a CSV file, and then ask questions about its content. The model will analyze the data summary to provide answers.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # File Uploader and DataFrame Display
                file_uploader = gr.File(
                    label="Upload your CSV",
                    file_types=[".csv"],
                    elem_id="file_uploader"
                )
                df_output = gr.DataFrame(
                    label="DataFrame Head",
                    headers=None,
                    wrap=True,
                    row_count=5, # Changed max_rows to row_count to fix the TypeError
                )
            
            with gr.Column(scale=2):
                # Chatbot Interface
                chatbot = gr.Chatbot(
                    label="Chatbox",
                    elem_classes="control-height"
                )
                query_box = gr.Textbox(
                    lines=3,
                    label="Your Question",
                    placeholder="e.g., How many rows are there? or What is the average value in the 'sales' column?"
                )

                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    regenerate_btn = gr.Button("Regenerate")
                    clear_btn = gr.Button("Clear History")

        # --- Event Handlers ---

        def load_csv_data(file, chatbot_history):
            """
            Handles the CSV file upload. It reads the CSV into a pandas DataFrame,
            stores it in the state, and displays its head in the UI.
            """
            if file is not None:
                try:
                    df = pd.read_csv(file.name)
                    chatbot_history.append((
                        f"Successfully loaded `{file.name}`.",
                        "I'm ready for your questions about the data."
                    ))
                    return df, df.head(), chatbot_history
                except Exception as e:
                    error_message = f"Error loading CSV: {e}"
                    chatbot_history.append((f"Failed to load `{file.name}`.", error_message))
                    return None, None, chatbot_history
            return None, None, chatbot_history

        def predict(query, chatbot_history, df):
            """
            Handles the user's query. It provides the model with context about
            the DataFrame before asking for a response.
            """
            if df is None:
                chatbot_history.append((query, "Please upload a CSV file first so I can answer your questions about it."))
                yield chatbot_history
                return

            # Create a context string with DataFrame info
            string_buffer = io.StringIO()
            df.info(buf=string_buffer)
            df_info = string_buffer.getvalue()
            
            contextual_query = f"""
You are an intelligent data analyst. Based on the following summary of a pandas DataFrame, please answer the user's question.

**DataFrame Head:**
```
{df.head().to_string()}
```

**DataFrame Info:**
```
{df_info}
```

---
**User Question:** {query}
"""
            chatbot_history.append((query, ""))
            full_response = ""
            for new_text in chat_stream(model, tokenizer, contextual_query, history=task_history.value):
                full_response += new_text
                chatbot_history[-1] = (query, full_response)
                yield chatbot_history
            
            # Update the task history after the full response is generated
            task_history.value.append((query, full_response))

        def regenerate(chatbot_history, task_history_state, df):
            """Regenerates the last response."""
            if not task_history_state:
                yield chatbot_history
                return
            
            last_item = task_history_state.pop(-1)
            chatbot_history.pop(-1)
            
            yield from predict(last_item[0], chatbot_history, df)

        def clear_history(chatbot, df_state, task_history_state, df_output):
            """Clears the chat history, DataFrame state, and UI."""
            task_history_state.clear()
            chatbot.clear()
            # Garbage collect to free up memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [], None, [], None

        def reset_user_input():
            """Clears the user input box."""
            return gr.update(value="")

        # Wire up the components to the functions
        file_uploader.upload(
            load_csv_data,
            inputs=[file_uploader, chatbot],
            outputs=[df_state, df_output, chatbot]
        )
        
        submit_btn.click(
            predict,
            inputs=[query_box, chatbot, df_state],
            outputs=[chatbot]
        ).then(reset_user_input, [], [query_box])

        regenerate_btn.click(
            regenerate,
            inputs=[chatbot, task_history, df_state],
            outputs=[chatbot]
        )

        clear_btn.click(
            clear_history,
            inputs=[chatbot, df_state, task_history, df_output],
            outputs=[chatbot, df_state, task_history, df_output]
        )
        
    return demo

def main():
    """The main function to load the model and launch the demo."""
    model, tokenizer = load_model_tokenizer()
    demo = build_chatbot_ui(model, tokenizer)
    
    args = _get_args()
    # Launch the Gradio app
    #demo.queue().launch(share=True, inbrowser=True)
    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )

if __name__ == "__main__":
    main()
