def load_css(fpath: str) -> str:
    with open(fpath) as f:
        return f.read()
