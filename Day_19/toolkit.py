def list_tools():
    return ["Knife", "Whisk", "Spatula", "Tongs"]

def recommend_tool(task):
    if task == "flip":
        return "Use a Spatula!"
    elif task == "stir":
        return "Use a Whisk!"
    else:
        return "Try using a Knife or Tongs."

def main():
    print("Welcome to the Chef's Toolkit!")
    print("Tools available:", list_tools())
    print(recommend_tool("stir"))

if __name__ == "__main__":
    main()
