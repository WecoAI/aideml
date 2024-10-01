import os

def print_directory_tree(startpath):
    """Prints the tree structure of the directory."""
    for root, dirs, files in os.walk(startpath):
        # Calculate the depth of the current directory
        depth = root.count(os.sep) - startpath.count(os.sep)
        indent = ' ' * 4 * depth  # 4 spaces for each depth level
        print(f"{indent}{os.path.basename(root)}/")  # Print directory name
        
        for f in files:
            print(f"{indent}    {f}")  # Print file names with indent

if __name__ == "__main__":
    project_directory = "."  # Change to the path of your project directory
    print_directory_tree(project_directory)
