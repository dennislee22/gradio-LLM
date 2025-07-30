# LLM Chatbot with Gradio

![mirazhi-demo](https://github.com/user-attachments/assets/447bdd44-73ff-42ee-a179-b26be2ce8db4)

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

5. Specify the LLM of your choice in [app-gradio.py](app-gradio.py).
Example:
  ```
  CHECKPOINT_PATH = "Llama-3.1-8B"
  ```

6. Specify if this LLM is running using cpu or cuda.
  ```
  device_map = "cpu"
  ```
 
7. Create CAI application and specify the frontend script [run-gradio.py](run-gradio.py). If the LLM is using CPU, ensure the RAM size in the profile is sufficient to fit the model.
<img width="456" height="734" alt="image" src="https://github.com/user-attachments/assets/69ba8fb5-3b7e-4953-91f7-728f561332f3" />

8. Start the application. Browse the exposed URL endpoint.

  When LLM uses CPU, it will make use of all the available CPU cores on the hosting node.

