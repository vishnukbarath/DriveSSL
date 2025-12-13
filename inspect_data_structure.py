import os

ROOT_DIR = r"C:\Users\vishn\Documents\DriveSSL\data"

def print_dir_tree(path, indent=""):
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return

    dirs = [d for d in entries if os.path.isdir(os.path.join(path, d))]

    for idx, d in enumerate(dirs):
        is_last = idx == len(dirs) - 1
        prefix = "└── " if is_last else "├── "
        print(f"{indent}{prefix}{d}/")

        next_indent = indent + ("    " if is_last else "│   ")
        print_dir_tree(os.path.join(path, d), next_indent)

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"[ERROR] Path does not exist: {ROOT_DIR}")
        return

    print(f"\nProject directory structure for:\n{ROOT_DIR}\n")
    print_dir_tree(ROOT_DIR)

if __name__ == "__main__":
    main()
