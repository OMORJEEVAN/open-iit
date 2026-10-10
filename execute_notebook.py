import nbformat
from nbclient import NotebookClient

def execute():
    nb_path = "CreditNirvana_PS2_Solution.ipynb"
    print(f"Reading notebook: {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    print("Executing notebook cells...")
    client.execute()

    print("Saving executed notebook...")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook executed and saved successfully!")

if __name__ == "__main__":
    execute()
