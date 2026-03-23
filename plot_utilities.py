import matplotlib


def get_font_size():
    return matplotlib.rcParams["font.size"]


def set_font_size(size):
    matplotlib.rcParams.update({"font.size": size})


class FontSize:
    def __init__(self, size):
        self.size = size
        self.original_size = get_font_size()

    def __enter__(self):
        set_font_size(self.size)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_font_size(self.original_size)
