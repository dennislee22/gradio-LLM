import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import sys
from io import StringIO
from argparse import ArgumentParser

# --- Global Configuration ---
CHECKPOINT_PATH = "Qwen2.5-7B-Instruct-1M"

def _get_args():
    """Parses command-line arguments for the web demo."""
    parser = ArgumentParser(description="Conversational AI Data Analyst Demo")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=CHECKPOINT_PATH,
        help="Hugging Face model checkpoint name or path, default: %(default)r",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the Gradio interface.",
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

def load_model_tokenizer(checkpoint_path):
    """Loads the pre-trained model and tokenizer from the specified path."""
    print(f"Loading model and tokenizer from {checkpoint_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        resume_download=True,
    )

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    ).eval()
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

# --- Core Chat Logic ---
def chat_stream(model, tokenizer, query, history):
    """Generates a response from the model in a streaming fashion."""
    messages = history + [{"role": "user", "content": query}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

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
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

# --- Main Application UI and Logic ---
def build_chatbot_ui(model, tokenizer):
    """Builds the Gradio web interface for the chatbot."""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".control-height { height: 500px; overflow: auto; }"
    ) as demo:
        df_state = gr.State(None)
        task_history = gr.State([])

        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1>🤖 Conversational AI Data Analyst</h1>
                <p>Upload a CSV, ask a question, and the AI will analyze the data and explain the results conversationally.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_uploader = gr.File(label="Upload your CSV", file_types=[".csv"])
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chatbox", elem_classes="control-height", type="messages")
                query_box = gr.Textbox(lines=3, label="Your Question", placeholder="e.g., What percentage of conversations were on each topic?")
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    regenerate_btn = gr.Button("Regenerate")
                    clear_btn = gr.Button("Clear History")

        def load_csv_data(file):
            if file is not None:
                try:
                    df = pd.read_csv(file.name)
                    history = [{"role": "assistant", "content": "✅ File loaded successfully. What would you like to know about the data?"}]
                    return df, history
                except Exception as e:
                    history = [{"role": "assistant", "content": f"❌ Error: {e}"}]
                    return None, history
            return None, []

        def predict(query, history, df):
            if df is None:
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": "Please upload a CSV file first."})
                yield history, history
                return

            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": "🤔 Step 1: Analyzing data..."})
            yield history, history

            string_buffer = StringIO()
            df.info(buf=string_buffer)
            df_info = string_buffer.getvalue()

            code_generation_prompt = f"**Schema (df.info()):**\n```{df_info}```\n**User Question:** \"{query}\""
            full_prompt = f"You are a data analysis AI. Based on the user's question and the DataFrame schema, write a Python script using pandas to get the data. The DataFrame is `df`. Your script must print the result. Provide only raw Python code.\n\n{code_generation_prompt}"
            
            text = tokenizer.apply_chat_template([{"role": "user", "content": full_prompt}], tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024, do_sample=False)
            response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            generated_code = response_text.split(full_prompt)[-1].strip()
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()

            try:
                old_stdout = sys.stdout
                redirected_output = sys.stdout = StringIO()
                exec(generated_code, {'df': df, 'pd': pd})
                sys.stdout = old_stdout
                raw_result = redirected_output.getvalue()

                history[-1] = {"role": "assistant", "content": "✍️ Step 2: Summarizing results..."}
                yield history, history
                
                if not raw_result:
                    summary = "The analysis ran successfully but produced no specific data to summarize."
                    history[-1] = {"role": "assistant", "content": summary}
                else:
                    summarization_prompt = f"You are a helpful AI assistant. Explain the following data to a user in a clear, conversational way. The user originally asked: \"{query}\"\nSummarize the key findings from the data below to answer their question.\n\n**Data to Summarize:**\n```{raw_result}```"
                    
                    final_summary = ""
                    for new_text in chat_stream(model, tokenizer, summarization_prompt, history=[]):
                        final_summary += new_text
                        history[-1] = {"role": "assistant", "content": final_summary}
                        yield history, history
            
            except Exception as e:
                sys.stdout = old_stdout
                error_message = f"❌ **Error during code execution:**\n\n```\n{str(e)}\n```"
                history[-1] = {"role": "assistant", "content": error_message}
                yield history, history

        def regenerate(history, df):
            if len(history) >= 2:
                history.pop()
                last_query_entry = history.pop()
                last_query = last_query_entry['content']
                yield from predict(last_query, history, df)
            else:
                yield history, history

        def clear_history():
            return [], None, []

        def reset_user_input():
            return gr.update(value="")

        # Wire up components
        file_uploader.upload(load_csv_data, [file_uploader], [df_state, chatbot])
        submit_btn.click(predict, [query_box, chatbot, df_state], [chatbot, task_history]).then(reset_user_input, [], [query_box])
        regenerate_btn.click(regenerate, [task_history, df_state], [chatbot, task_history])
        clear_btn.click(clear_history, [], [chatbot, df_state, task_history])
        
    return demo

def main():
    """The main function to parse arguments, load the model, and launch the demo."""
    args = _get_args()
    
    model, tokenizer = load_model_tokenizer(args.checkpoint_path)
    demo = build_chatbot_ui(model, tokenizer)
    
    # Launch the Gradio app with arguments from the command line
    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )

if __name__ == "__main__":
    main()