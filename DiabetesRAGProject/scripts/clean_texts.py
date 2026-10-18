import os, re

DIR = "../data/kb"
for file in os.listdir(DIR):
    if file.endswith(".txt"):
        path = os.path.join(DIR, file)
        text = open(path).read()
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[^a-zA-Z0-9.,;:\-\n ]", "", text)
        open(path, "w").write(text.strip())
        print(f"✅ Cleaned {file}")
