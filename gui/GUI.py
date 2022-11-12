import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image, ImageTk

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

class ImageCanvas(tk.Canvas):
    def __init__(self, container):
        super().__init__(container)
        self.load_image('./gui/images/test.jpg')

    def load_image(self, path):
        self.image = Image.open(path)
        self.img_copy= self.image.copy()
        self.photo = ImageTk.PhotoImage(self.image)

class MainFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.widget_setup()

    def widget_setup(self):
        image_canvas = ImageCanvas(self)
        image_canvas.pack(expand=True, fill=tk.BOTH, side=tk.LEFT)
        image_canvas.bind('<Configure>', self.action_resize_image)
        self.image_canvas = image_canvas

        button_frame = ButtonFrame(self, MainFrame.action_event, self.action_load_image)
        button_frame.pack(fill=tk.BOTH, side=tk.LEFT)

    def action_event(event = None):
        print("Event happened")

    def action_load_image(self, event):
        filename = fd.askopenfilename()
        self.image_canvas.load_image(filename)
        self.image_canvas.update()


    def action_resize_image(self, event):
        canvas = event.widget
        origin = (0,0)
        size = (event.width, event.height)
        if canvas.bbox("bg") != origin + size:
            canvas.delete("bg")
            canvas.image = canvas.img_copy.resize(size)
            canvas.photo = ImageTk.PhotoImage(canvas.image)
            canvas.create_image(*origin,anchor="nw",image=canvas.photo,tags="bg")
            canvas.tag_lower("bg","all")

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
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)

        # set the position of the window to the center of the screen
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.iconbitmap('./gui/icons/app_icon.ico')
        
    def mainloop(self):
        frame = MainFrame(self)
        frame.pack(fill=tk.BOTH, expand=True)

        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        finally:
            super().mainloop()
        



if __name__ == "__main__":
    gui = EstimatorApp()
    gui.mainloop()


