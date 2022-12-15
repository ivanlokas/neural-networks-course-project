from pathlib import Path

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

import torch

from PIL import Image, ImageTk
from torchvision.transforms import transforms

from datasets.load import CustomImageFolder
from models.cnn_deep import DeepModel


class EstimatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Age estimator")

        # window size
        window_width = 700
        window_height = 700

        # screen dimension
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # center coordinates
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)

        # set the position of the window to the center of the screen
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.iconbitmap(Path(__file__).parent / 'icons/app_icon.ico')

    def mainloop(self):
        frame = MainFrame(self)
        frame.pack(fill=tk.BOTH, expand=True)

        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        finally:
            super().mainloop()


class MainFrame(ttk.Frame):
    # Image transform
    transform = transforms.ToTensor()

    # Data
    data_path = Path(__file__).parent.parent / 'datasets' / 'UTKFace_grouped'
    data = CustomImageFolder(data_path, transform=transforms.ToTensor())

    # Model
    model = DeepModel()

    # Load state dict
    path = Path(__file__).parent.parent / 'states' / 'deep_bs_16_ne_100_lr_0.001_wd_1e-06_g_0.9999' / f'epoch_{50}'
    model.load_state_dict(torch.load(path))

    def __init__(self, container):
        super().__init__(container)
        self.widget_setup()
        self.style_setup()

    def style_setup(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure("TButton", font=("Helvetica", 16),
                    foreground="#F7FBEF",
                    background="#292929",
                    lightcolor="none",
                    darkcolor="none",
                    focuscolor="#333333",
                    borderwidth=0,
                    bordercolor="none")

        s.configure("TLabel", font=("Helvetica", 64),
                    foreground="#F7FBEF",
                    background="#292929",
                    lightcolor="none",
                    darkcolor="none",
                    focuscolor="#333333",
                    borderwidth=0,
                    bordercolor="none")

        s.configure("TFrame",
                    foreground="#F7FBEF",
                    background="#1F1F1F",
                    lightcolor="none",
                    darkcolor="none",
                    focuscolor="#333333",
                    borderwidth=0,
                    bordercolor="none")

        s.map('TButton',
              background=[("pressed", "#292929"),
                          ("active", "#333333")],
              borderwidth=[("active", 0)],
              bordercolor=[("active", "none")],
              lightcolor=[("active", "none")],
              darkcolor=[("active", "none")],
              foreground=[("pressed", "#F7FBEF"),
                          ("active", "#F7FBEF")]
              )

    def widget_setup(self):
        image_canvas = ImageCanvas(self)
        image_canvas.pack(expand=True, fill=tk.BOTH, side=tk.LEFT)
        image_canvas.bind('<Configure>', self.action_resize_image)
        self.image_canvas: ImageCanvas = image_canvas

        button_frame: ButtonFrame = ButtonFrame(
            self, self.action_estimate, self.action_load_image)
        button_frame.pack(fill=tk.BOTH, side=tk.TOP)

        result_label = ttk.Label(self, text="-", anchor="center")
        result_label.pack(ipadx=10, ipady=30, fill=tk.BOTH, side=tk.BOTTOM)
        self.result_label = result_label

    # For testing purposes only
    def action_event(self, event=None):
        print("Event happened")

    def action_estimate(self, event=None):
        path = self.image_canvas.image_path

        custom_image = MainFrame.data.loader(path)
        custom_image = MainFrame.transform(custom_image)
        custom_image = torch.unsqueeze(custom_image, 0)

        model_output = MainFrame.model(custom_image)
        model_output = round(model_output.item())

        self.result_label["text"] = model_output

    def action_load_image(self, event):
        filename = fd.askopenfilename()
        if not filename:
            return

        self.image_canvas.load_image(filename)
        self.image_canvas.update()

    def action_resize_image(self, event):
        canvas = event.widget
        origin = (0, 0)
        size = (event.width, event.height)
        if canvas.bbox("bg") != origin + size:
            canvas.display_image(origin, size)


class ImageCanvas(tk.Canvas):
    def __init__(self, container):
        super().__init__(container)
        self.load_image(Path(__file__).parent / 'images/test.jpg')
        self["highlightthickness"] = 0

    def load_image(self, path):
        self.image_path = path
        self.image = Image.open(path)
        self.img_copy = self.image.copy()
        self.photo = ImageTk.PhotoImage(self.image)
        self.display_image()

    def display_image(self, origin=(0, 0), size=None):
        if size is None:
            size = (self.winfo_width(), self.winfo_height())

        self.delete("bg")
        self.image = self.img_copy.resize(size)
        self.photo = ImageTk.PhotoImage(self.image)
        self.create_image(*origin, anchor="nw", image=self.photo, tags="bg")
        self.tag_lower("bg", "all")


class ButtonFrame(ttk.Frame):
    def __init__(self, container, action1, action2):
        super().__init__(container)
        self.widget_setup(action1, action2)

    def widget_setup(self, action1, action2):
        ipadding = {'ipadx': 10, 'ipady': 10}

        btn = ttk.Button(self, text='Estimate')
        btn.bind('<Button-1>', action1, add="+")
        btn.pack(**ipadding, fill=tk.X)
        btn = ttk.Button(self, text='Load image')
        btn.bind('<Button-1>', action2, add="+")
        btn.pack(**ipadding, fill=tk.X)


if __name__ == "__main__":
    gui = EstimatorApp()
    gui.mainloop()
