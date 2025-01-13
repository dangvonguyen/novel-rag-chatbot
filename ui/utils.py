def load_css(fpath: str) -> str:
    with open(fpath, "r") as f:
        return f.read()
