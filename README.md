# LLM Chatbot with Gradio

![gradio-LLM](https://github.com/user-attachments/assets/8b309f74-4781-4ea4-85e0-92a0898fbbfc)


## Platform Requirement
☑️ Python 3.11

☑️ Cloudera AI(CAI)/Cloudera Machine Learning (CML) 1.5.x

## Procedure

1. Create a new CAI project.
   
2. Install python libraries.
  ```
  pip install accelerate torch transformers gradio ipywidgets pandas numpy bpytop
  ```

3. Download the LLM into the project of the CAI/CML platform using either `git clone` or `wget`.
Example:
  ```
  git lfs clone https://huggingface.co/meta-llama/Llama-3.1-8B
  ```

4. Ensure [run-gradio.py](run-gradio.py) and [app-gradio.py](app-gradio.py) scripts are in the CAI project.

5. Specify the LLM of your choice in [run-gradio.py](run-gradio.py). Example: Llama-3.1-8B
Example:
  ```
  os.system("python app-gradio.py --server-name=127.0.0.1 --checkpoint-path=Llama-3.1-8B --server-port=$CDSW_APP_PORT > gradio.log 2>&1")
  ```

6. Specify if LLM is running using cpu or cuda in [app-gradio.py](app-gradio.py).
  ```
  device_map = "cpu"
  ```
 
7. Create CAI application and specify the frontend script [run-gradio.py](run-gradio.py). If the LLM is using CPU, ensure the RAM size in the profile is sufficient to fit the model.
<img width="456" height="734" alt="image" src="https://github.com/user-attachments/assets/69ba8fb5-3b7e-4953-91f7-728f561332f3" />

8. Start the application. Browse the exposed URL endpoint.
<img width="669" height="318" alt="image" src="https://github.com/user-attachments/assets/82646a28-c8d6-4f10-95a4-9bb84389289a" />

<img width="669" height="700" alt="image" src="https://github.com/user-attachments/assets/e5ef889a-f491-45fe-878a-fa299b566aef" />

🔺 When LLM uses CPU, it will make use of all the available CPU cores on the hosting node.

![bpytop-llm](https://github.com/user-attachments/assets/74778045-b1a4-4ea3-a602-e7e86cc10d07)

🔺 Memory consumption:

<img width="700" height="587" alt="image" src="https://github.com/user-attachments/assets/e8fdf550-e413-4396-ba0a-63cc10e1a9fb" />



