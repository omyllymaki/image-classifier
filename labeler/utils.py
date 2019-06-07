from tkinter import Tk, filedialog


def initialize_tkinter():
    root = Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)


def open_files_dialog():
    initialize_tkinter()
    paths = filedialog.askopenfilename(multiple=True)
    return paths


def save_file_dialog():
    initialize_tkinter()
    path = filedialog.asksaveasfilename(initialfile='labels',
                                        filetypes=[('csv', '.csv'), ('all files', '.*')],
                                        defaultextension='.csv', )
    return path
