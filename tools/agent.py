from .interpreter import CodeInterpreter
from typing import Callable
from tqdm import tqdm 
from .repo import *
from .podcast import *

def strategic_respond(json_file_path: str, 
                      img_base64: str,
                      user_question: str,
                      file_dag: dict,
                      interpreter: CodeInterpreter, 
                      get_claude_response: Callable):
    
    retrieved_info = ""
    progress_bar = tqdm(total=2, desc="Strategic response progress", unit="%")
    
    print("Coding to traverse file dependency graph...")
    prompt = get_navigate_prompt(json_file_path, user_question, retrieved_info, file_dag)
    response = get_claude_response(prompt, img=img_base64, img_type="image/png")
    code_result, _, _ = interpreter(response)
    print(f"Compiled code result: {code_result}")
    
    progress_bar.update(1)
    
    print("Forming final answer...")
    prompt = get_navigate_prompt(json_file_path, user_question, code_result, file_dag, use_code=False)
    response = get_claude_response(prompt, img=img_base64, img_type="image/png")
    final_response, figure, _ = interpreter(response)
    
    progress_bar.update(1)
    
    progress_bar.close()
    
    return final_response

def dag_to_prompt(dag: dict, no_file_content: bool = False):
    """
    Convert a dependency graph (DAG) to a prompt for analysis, including file contents.
    
    Args:
    dag (dict): A dictionary representing the dependency graph.
    no_file_content (bool): If True, skip including file contents in the prompt.
    
    Returns:
    str: A prompt for analyzing the codebase structure and contents.
    """
    prompt = "Analyze the following codebase structure, dependencies, and file contents:\n\n"
    
    # Add information about nodes (files) and their contents
    for node, data in dag.items():
        if data['type'] == "file" and no_file_content:
            continue
        prompt += f"File: {data['name']}\n"
        content_str = read_node_content(data)
        prompt += f"Content:\n```\n{content_str}\n```\n\n"
        
        if 'dependencies' in data:
            prompt += "Dependencies:\n"
            for dep in data['dependencies']:
                prompt += f"  - {dag[dep]['name']}\n"
        
        prompt += "\n" + "-"*50 + "\n\n"
    
    prompt += "Instructions for analysis:\n"
    prompt += "1. Examine each file's content and its role in the project.\n"
    prompt += "2. Analyze the dependencies between files to understand the code flow.\n"
    prompt += "3. Identify any central or frequently depended-upon files that might be core to the project.\n"
    prompt += "4. Look for patterns in the file naming and organization that might indicate the project's architecture.\n"
    prompt += "5. Based on the file contents and dependencies, determine the main functionality of the project.\n"
    prompt += "6. Suggest any potential areas for improvement in the code organization or dependency structure.\n"
    prompt += "7. Provide a summary of your findings, including the project's purpose and key features.\n"

    return prompt
    
    
class RepoAgent: # Pure Text-Based 
    
    def __init__(self, temp_repo: str, get_vlm_response: Optional[Callable] = None, sandbox_dir = "sandbox"):
        self.temp_repo = temp_repo
        self.get_vlm_response = get_vlm_response
        self.sandbox_dir = sandbox_dir
        file_dag = directory_to_file_dag(temp_repo)
        file_dag = assign_importance_score_to_dag(file_dag)
        file_dag = assign_levels(file_dag)
        file_dag = set_full_opacity(file_dag)
        self.file_dag = file_dag
        self.file_names = [self.file_dag[k]['name'] for k in self.file_dag]
        module_dag = parse_directory_dag(temp_repo)
        module_dag = assign_importance_score_to_dag(module_dag)
        module_dag = assign_levels(module_dag)
        self.module_dag = module_dag
        # self._summarize_code()
        self.module_names = [self.module_dag[k]['name'] for k in self.module_dag]
        self.module_dag_path = save_dag_as_json(self.module_dag, temp_repo, self.sandbox_dir)
        
    def _summarize_code(self):
        if not check_dag_exists(self.module_dag, self.temp_repo) and self.get_vlm_response:
            self.module_dag = bottom_up_summarization(self.module_dag, self.get_vlm_response) # Get a nice VLM to speed-up the inference, in case claude dies ...
            save_dag_as_json(self.module_dag, self.temp_repo)
    
    def visualize_file(self):
        name = self.temp_repo.split("/")[-1]
        return visualize_dag(self.file_dag, name=name, output_dir=self.sandbox_dir)
    
    def show_files(self):
        python_file_names = get_python_files(self.temp_repo)
        present_str = f"Python Files in repo: {self.temp_repo}"
        for i, file_name in enumerate(python_file_names, 1):
            present_str += f"\n{i}. {file_name}"
        print(present_str)
    
    @property
    def python_files_dict(self): # useful for agent prompting
        python_file_dict = {}
        # List Python files
        python_file_names = get_python_files(self.temp_repo)
        # Print the list of Python file names with numbers
        present_str = f"Python Files in repo: {self.temp_repo}"
        for i, file_name in enumerate(python_file_names, 1):
            present_str += f"\n{i}. {file_name}"
            python_file_dict[i] = file_name
        # print(present_str)
        return python_file_dict

    def get_module_name(self, module_name_or_number: str):
        if isinstance(module_name_or_number, int):
            return self.module_names[module_name_or_number - 1]
        else:
            return module_name_or_number
        
    def get_module_dag(self, module_name_or_number: str, cap_node_number: int = 40, depth: int = 6):
        module_name = self.get_module_name(module_name_or_number)
        dag = parse_file_dag(self.temp_repo, module_name, depth=depth)            
        dag = decide_opacity_of_dag(dag, cap_node_number=cap_node_number)
        dag = assign_levels(dag)
        return dag
            
    def visualize_module(self, module_name_or_number: str, cap_node_number: int = 40, depth: int = 6):
        dag = self.get_module_dag(module_name_or_number, cap_node_number=cap_node_number, depth=depth)
        name = self.temp_repo.split("/")[-1] + "_" + dag[list(dag.keys())[0]]['name']
        return visualize_dag(dag, cap_node_number = cap_node_number, name=name, output_dir=self.sandbox_dir)
    
    def animate_module(self, module_name_or_number: str, frame_count: int = 50, fps: int = 10, cap_node_number: int = 40, depth: int = 6):
        dag = self.get_module_dag(module_name_or_number, cap_node_number=cap_node_number, depth=depth)
        name = self.temp_repo.split("/")[-1] + "_" + dag[list(dag.keys())[0]]['name']
        output_file = create_gif_from_dag(dag, output_name = f"anime_{name}", frame_count=frame_count, fps=fps)
        print("GIF file saved in path: ", output_file)

    def animate_file(self, frame_count, fps):
        name = self.temp_repo.split("/")[-1]
        output_file = create_gif_from_dag(self.file_dag, output_name = f"anime_{name}", fps=fps, frame_count=frame_count)
        print("GIF file saved in path: ", output_file)
        
    @property 
    def module_graph(self):
        png_module_graph_path = self.visualize_module()
        return file_to_preprocessed_img(png_module_graph_path)
    
    @property 
    def file_graph(self):
        png_file_graph_path = self.visualize_file()
        return file_to_preprocessed_img(png_file_graph_path)
    
    def _respond(self, question: str):
        if not self.get_vlm_response:
            raise ValueError("Get VLM response function not provided yet")
        interpreter = CodeInterpreter()
        return strategic_respond(self.module_dag_path, self.module_graph, question, self.module_dag, interpreter, self.get_vlm_response) # Module DAG contains File DAG
    
    def generate_podcast(self, module_name_or_number: Optional[str] = None):
        raw_script = write_podcast_script(prompt=self._to_prompt(module_name_or_number))
        script = parse_script(raw_script)
        if not module_name_or_number:
            name = self.temp_repo.split("/")[-1]
        else:
            name = self.temp_repo.split("/")[-1] + "_" + module_name_or_number.split("/")[-1].replace(".py", "")
        generate_podcast_audio(script, name=name, output_dir=self.sandbox_dir)
        
    
    def _to_prompt(self, module_name_or_number: Optional[str]=None):
        if not module_name_or_number:
            prompt = dag_to_prompt(self.file_dag)        
        else:
            prompt = dag_to_prompt(self.get_module_dag(module_name_or_number), no_file_content=True)
        return prompt
