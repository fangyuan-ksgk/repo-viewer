import os
import subprocess
import numpy as np
from PIL import Image
import time
import copy
import base64
import io
from tqdm import tqdm

d2_prefix = """vars: {
  d2-config: {
    sketch: true
  }
}
classes: {
  file: {
    label: ""
    shape: diamond
    style: {
      fill: yellow
      shadow: true
    }
  }
}

classes: {
  class: {
    label: ""
    shape: hexagon
    style: {
      fill: lightblue
      shadow: true
    }
  }
}

classes: {
  function: {
    label: ""
    shape: rectangle
    style: {
      fill: white
      shadow: false
    }
  }
}"""






object_template_with_overhead = """{object_name}.class: {object_type}
{object_name}.label: "{object_label}"
{object_name}: {{
  style: {{
    opacity: {opacity}
    stroke: "black"
    stroke-width: 4
    shadow: true
  }}
}}"""

object_template = """{object_name}.class: {object_type}
{object_name}.label: "{object_label}"
{object_name}: {{
  style: {{
    opacity: {opacity}
    stroke: "black"
    stroke-width: 4
    shadow: true
  }}
}}"""

# used for file-level dependency parsing
get_parent_file = lambda node: node['file'].replace(".py", "").replace("/",".")
get_object_label = lambda node: node['name'].split("::")[-1].replace(".py", "")

# used for function-level dependency parsing | avoid name conflict between file and function (main.py -- main)
_get_parent_file = lambda node: node['file'].replace(".py", "").replace("/",".")
_get_object_label = lambda node: node['name'].split("::")[-1].replace(".py", "_")

def build_d2_node(node: dict, node_id: str, include_overhead: bool = False, file_level: bool = True) -> str:
    if file_level:
        parent_file = get_parent_file(node)
        object_label = get_object_label(node)
    else:
        parent_file = _get_parent_file(node)
        object_label = _get_object_label(node)
    if include_overhead:
        object_name = f"{parent_file}.{object_label}"  
    else:
        object_name = object_label 
    object_type = node["type"]
    if object_type not in ["file", "class"]:
        object_type = "function"
    opacity = node["opacity"]
    opacity = min(1.0, max(0.0, opacity))
    opacity_str = f"{opacity:.2f}"
    
    return object_template.format(object_name=object_name, object_type=object_type, object_label=object_label, opacity=opacity_str)

# Would love to create more caveat line visual
link_template = """{start_object_id} -> {end_object_id}: {{
  style.stroke: black
  style.opacity: {opacity}
  style.stroke-width: 2
}}"""

link_file_template = """{start_object_id} -> {end_object_id}: {{
  style.stroke: red
  style.opacity: {opacity}
  style.stroke-width: 2
  style.stroke-dash: 5
  style.animated: true
}}"""

def get_object_name(node: dict, file_level: bool = True) -> str:
    if file_level:
        parent_file = get_parent_file(node)
        object_label = get_object_label(node)
        object_name = f"{parent_file}.{object_label}" 
    else:
        parent_file = _get_parent_file(node)
        object_label = _get_object_label(node)
        object_name = f"{parent_file}.{object_label}" 
    return object_name

def get_label_name(node: dict, file_level: bool = True) -> str:
    if file_level:
        return node['name'].split("::")[-1].replace(".py", "")
    else:
        return node['name'].split("::")[-1].replace(".", "_")

def build_d2_edge(str_node: dict, end_node: dict, include_overhead: bool = False, file_level: bool = True) -> str:
    opacity = min(1.0, max(0.0, end_node["opacity"]))
    opacity_str = f"{opacity:.2f}"
    
    if include_overhead:
        start_object_name = get_object_name(str_node, file_level=file_level)
        end_object_name = get_object_name(end_node, file_level=file_level)
    else:
        start_object_name = get_label_name(str_node, file_level=file_level)
        end_object_name = get_label_name(end_node, file_level=file_level)
    
    if str_node["type"] == "file" and end_node["type"] == "file":
        return link_file_template.format(start_object_id=start_object_name, end_object_id=end_object_name, opacity=opacity_str)
    if str_node["type"] == "file" and end_node["type"] != "file":
        return link_file_template.format(start_object_id=start_object_name, end_object_id=end_object_name, opacity=opacity_str)
    if str_node["type"] != "file":
        return link_template.format(start_object_id=start_object_name, end_object_id=end_object_name, opacity=opacity_str)


def build_d2_from_dag(dag: dict, include_overhead: bool = False, file_level: bool = True) -> str:
    """
    Convert Sub-DAG dictionary into d2 code
    """
    d2_code = d2_prefix 

    for node_id, node in dag.items():
        object_str = build_d2_node(node, node_id, include_overhead, file_level)
        d2_code += "\n" + object_str

    for node_id, node in dag.items():
        edge_pairs = [(node_id, end_node) for end_node in node['edges']]    
        for start, end in edge_pairs:
            link_str = build_d2_edge(dag[start], dag[end], include_overhead, file_level)
            if link_str:
                d2_code += f"\n{link_str}"
            
    return d2_code


def save_png_from_d2(d2_code, file_name, output_dir="d2_output"):
    """
    Save the d2_code as a .d2 file and generate the corresponding .svg file.
    
    Args:
    d2_code (str): The D2 diagram code.
    file_name (str): The base name for the output files (without extension).
    output_dir (str): The directory to save the files in.
    
    Returns:
    str: The path to the saved PNG file, or None if an error occurred.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the .d2 file
    d2_file_path = os.path.join(output_dir, f"{file_name}.d2")
    with open(d2_file_path, "w") as f:
        f.write(d2_code)
    
    # Generate the PNG file using the d2 command-line tool
    png_file_path = os.path.join(output_dir, f"{file_name}.png")
    try:
        subprocess.run(["d2", d2_file_path, png_file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating PNG: {e}")
        png_file_path = None
    except FileNotFoundError:
        print("Error: d2 command not found. Make sure d2 is installed and in your PATH.")
        png_file_path = None
    
    return png_file_path


def visualize_dag(dag: dict, output_dir="d2_output", show: bool = True):
    """
    Visualize the DAG using d2
    """
    if 'opacity' not in dag[list(dag.keys())[0]]:
        dag = decide_opacity_of_dag(dag, progress=1.0, cap_node_number=30)
    d2_code = build_d2_from_dag(dag, include_overhead=True)
    png_file_path = save_png_from_d2(d2_code, "dag", output_dir=output_dir)
    if png_file_path:
        if show:    
            dag_graph = Image.open(png_file_path)
            dag_graph.show()
    else:
        print("Error: PNG file could not be generated.")
    
    return png_file_path
    
    
    
def save_svg_from_d2(d2_code, file_name, output_dir="d2_output"):
    """
    Save the d2_code as a .d2 file and generate the corresponding .svg file.
    
    Args:
    d2_code (str): The D2 diagram code.
    file_name (str): The base name for the output files (without extension).
    output_dir (str): The directory to save the files in.
    
    Returns:
    tuple: Paths to the saved .d2 and .svg files.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the .d2 file
    d2_file_path = os.path.join(output_dir, f"{file_name}.d2")
    with open(d2_file_path, "w") as f:
        f.write(d2_code)
    
    # Generate the .svg file using the d2 command-line tool
    svg_file_path = os.path.join(output_dir, f"{file_name}.svg")
    try:
      # Redirect stdout and stderr to devnull
        subprocess.run(["d2", d2_file_path, svg_file_path], check=True)
    except subprocess.CalledProcessError as e:
        svg_file_path = None
    except FileNotFoundError:
        print("Error: d2 command not found. Make sure d2 is installed and in your PATH.")
        svg_file_path = None
    
    return d2_file_path, svg_file_path


def filter_opacity_graph(graph):
    """ 
    Filter SubGraph, remove nodes and edges:  
    - Keep nodes with non-zero opacity
    - Remove nodes with zero opacity
    - Remove edges accordingly 
    """
    filtered_graph = {}
    for node_id, node_data in graph.items():
        if node_data['opacity'] > -1:
            filtered_graph[node_id] = node_data.copy()
            filtered_graph[node_id]['edges'] = [
                edge for edge in node_data['edges']
                if graph[edge]['opacity'] > 0
            ]
    return filtered_graph



def decide_opacity_of_dag(dag: dict, progress: float, cap_node_number: int = 15) -> dict:
    # Adjust importance scores based on hierarchy
    importance_groups = {}
    for node, data in dag.items():
        importance = data.get('importance', 0)
        file_path = data.get('file_path', '')
        level = file_path.count('/')
        adjusted_importance = importance + (1 / (level + 1))  # Adjust importance based on level
        if adjusted_importance not in importance_groups:
            importance_groups[adjusted_importance] = []
        importance_groups[adjusted_importance].append(node)

    # Sort nodes by adjusted importance
    sorted_nodes = sorted(dag.items(), key=lambda x: x[1]['importance'], reverse=True)

    # Calculate opacities
    scores = np.array([data['importance'] for (_, data) in sorted_nodes])
    opacities = scores / scores.max()
    
    # buffer period 
    bp = 0.2
    if progress < 0.2:
        max_opacity = opacities[opacities < 1.0].max() if len(opacities[opacities < 1.0]) > 0 else 0
        target_add_opacity = (1.0 - max_opacity) * bp 
        target_opacities = np.minimum(opacities + target_add_opacity, 1.0)
        begin_opacity = np.where(opacities < 1.0, 0.0, opacities)
        # interpolate between 
        interpolate_progress = progress * (1/bp)
        opacities = interpolate_progress * (target_opacities - begin_opacity) + begin_opacity
    else: 
        # Apply progress
        max_opacity = opacities[opacities < 1.0].max() if len(opacities[opacities < 1.0]) > 0 else 0
        add_opacity = (1.0 - max_opacity) * progress
        opacities = np.minimum(opacities + add_opacity, 1.0)

    # Cap the number of visible nodes
    opacities[cap_node_number:] = -1

    # Update the dag with new opacities
    for (node, data), opacity in zip(sorted_nodes, opacities):
        dag[node]['opacity'] = float(opacity)

    dag = filter_opacity_graph(dag)
    dag = assign_levels(dag)

    return dag


# SUDO Gif: level-wise appearing animation with D2-diagram

def cap_dag_count(dag: dict, cap_node_number: int = 15) -> dict:
    # Adjust importance scores based on hierarchy
    importance_groups = {}
    for node, data in dag.items():
        importance = data.get('importance', 0)
        file_path = data.get('file_path', '')
        level = file_path.count('/')
        adjusted_importance = importance + (1 / (level + 1))  # Adjust importance based on level
        if adjusted_importance not in importance_groups:
            importance_groups[adjusted_importance] = []
        importance_groups[adjusted_importance].append(node)

    # Sort nodes by adjusted importance
    sorted_nodes = sorted(dag.items(), key=lambda x: x[1]['importance'], reverse=True)
    
    # Assign 1.0 opacity to the first `cap_node_number` nodes, rest are set to -1
    opacities = np.ones(len(sorted_nodes))
    opacities[cap_node_number:] = -1

    # Update the dag with new opacities
    for (node, data), opacity in zip(sorted_nodes, opacities):
        dag[node]['opacity'] = float(opacity)

    return filter_opacity_graph(dag)


def assign_levels(sub_dag):
    # Create a dictionary to store incoming edges for each node
    incoming_edges = {node: set() for node in sub_dag}
    for node, data in sub_dag.items():
        for edge in data.get('edges', []):
            incoming_edges[edge].add(node)
    
    # Find nodes with no incoming edges (level 1)
    level = 1
    current_level_nodes = [node for node, edges in incoming_edges.items() if not edges]
    
    # Assign levels to all nodes
    while current_level_nodes:
        for node in current_level_nodes:
            sub_dag[node]['level'] = level
        
        # Find next level nodes
        next_level_nodes = []
        for node in sub_dag:
            if 'level' not in sub_dag[node]:
                if all(sub_dag.get(parent, {}).get('level') is not None for parent in incoming_edges[node]):
                    next_level_nodes.append(node)
        
        current_level_nodes = next_level_nodes
        level += 1
    
    return sub_dag


def generate_opacity_frames(sub_dag, frame_count, static_portion: float = 0.2):
    """ 
    Generate opacity frames for the DAG animation
    - Nodes appear gradually from top to bottom
    - At the end, all nodes will be fully visible for 'static_portion * frame_count' frames
    """
    # Reset opacity to zero for all nodes
    for node in sub_dag:
        sub_dag[node]['opacity'] = 0.0
    
    # Sort nodes by level and then by their order in the dictionary
    sorted_nodes = sorted(sub_dag.items(), key=lambda x: (x[1]['level'], list(sub_dag.keys()).index(x[0])))
    
    frames = []
    for frame in range(frame_count):
        current_frame_dag = copy.deepcopy(sub_dag)
        
        # Calculate the overall progress for this frame
        overall_progress = (frame + 1) / frame_count
        
        for i, (node_id, node_data) in enumerate(sorted_nodes):
            # Calculate the node's individual progress
            node_progress = i / (len(sorted_nodes) - 1)
            
            # If the overall progress has reached this node's turn to appear
            if overall_progress >= node_progress:
                # Calculate the node's opacity based on how long it's been visible
                node_opacity = min(1.0, (overall_progress - node_progress) / (1 / (len(sorted_nodes) - 1)))
                current_frame_dag[node_id]['opacity'] = node_opacity
            else:
                current_frame_dag[node_id]['opacity'] = 0.0
                    
                    
        # Check if we're in the last 20% of frames
        if frame >= frame_count * 0.8:
            # Calculate progress for the final interpolation
            final_progress = (frame - frame_count * 0.8) / (frame_count * 0.2)
            
            # Gradually increase opacity for all nodes to reach 1.0 at the end
            for node_id in current_frame_dag:
                current_opacity = current_frame_dag[node_id]['opacity']
                # Ensure we don't decrease opacity, only increase it
                new_opacity = max(current_opacity, min(1.0, current_opacity + (1.0 - current_opacity) * final_progress))
                current_frame_dag[node_id]['opacity'] = new_opacity            
        
        frames.append(current_frame_dag)
        
    # Add static frames at the end
    for _ in range(int(frame_count * static_portion)):
        frames.append(copy.deepcopy(current_frame_dag))
    
    return frames
  

def create_gif(png_files: list, output_file: str = "commit_dag_evolution.gif", fps: int = 2):
    # Define a common size for all frames
    MAX_SIZE = (1024, 512)  # You can adjust this as needed

    # Create GIF
    images = []
    for png_file in tqdm(png_files, desc="Creating GIF"):
        if os.path.exists(png_file):
            # Open the image
            img = Image.open(png_file)
            
            # Resize the image while maintaining aspect ratio
            img.thumbnail(MAX_SIZE, Image.LANCZOS)
            
            # Create a new image with white background
            new_img = Image.new("RGB", MAX_SIZE, (255, 255, 255))
            
            # Paste the resized image onto the center of the new image
            new_img.paste(img, ((MAX_SIZE[0] - img.size[0]) // 2,
                                (MAX_SIZE[1] - img.size[1]) // 2))
            
            images.append(new_img)

    if images:
        # Calculate duration based on fps
        duration = int(1000 / fps)  # Convert fps to milliseconds between frames
        images[0].save(output_file, save_all=True, append_images=images[1:], 
                    duration=duration, loop=0)
        # print(f"Animation saved as {output_file}")
    else:
        print("No PNG files were found to create the GIF.")
    
    
def write_dependency_dags(dags, output_dir="d2_output", n_frames=10):
    
    pbar = tqdm(total=len(dags)*n_frames, desc="Processing DAGs into PNG frames")

    for i, dag in enumerate(dags):
        
        # Animate the growing process by increasing progress from 0 to 1
        for progress in np.linspace(0, 1, n_frames):
            sub_dag = decide_opacity_of_dag(dag, progress=progress, cap_node_number=15)
            d2_code = build_d2_from_dag(sub_dag, include_overhead=True)
            
            # Save each frame as an SVG
            filename = f"commit_{i}_progress_{progress:.2f}"
            save_png_from_d2(d2_code, filename, output_dir=output_dir)
            
            time.sleep(0.5)  # Add a small delay between frames
            
            pbar.update(1)  # Update progress bar

    pbar.close()  # Close the progress bar when done
    
    
def file_to_preprocessed_img(file_path):
    
    if file_path.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file_path)
    else:
        raise ValueError("Unknown file format")

    # Convert image to PNG
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    
    # Convert PNG to base64
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_base64