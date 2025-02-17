import tkinter as tk
import cv2
import numpy as np
from PIL import ImageTk, Image

class TkinterUtils:
    def __init__(self, img1, img2, width = None, height = None):
        # Initialize window
        self.root = tk.Tk()

        # Configure window
        self.root.title("Assignment 2 -> Lang Towl")

        if width and height:
            self.root.geometry(f"{width}x{height}")

        # Pre-process passed images
        self.image1 = Image.open(img1)
        self.image1_copy = Image.open(img1)
        self.image2 = Image.open(img2)
        self.image2_copy = Image.open(img2)

        # Convert images to Tkinter-compatible format
        self.tk_image1 = ImageTk.PhotoImage(self.image1)
        self.tk_image1_copy = ImageTk.PhotoImage(self.image1_copy)
        self.tk_image2 = ImageTk.PhotoImage(self.image2)
        self.tk_image2_copy = ImageTk.PhotoImage(self.image2_copy)

        # Render canvas
        self.update_canvas()

        # Add UI elements
        self.button = tk.Button(self.root, text="Press Me", command=self.on_button_pressed)
        self.button.grid(row=2, column=0, columnspan=2)

        # Start main event loop
        self.root.mainloop()

    def update_canvas(self):
        # Convert images to Tkinter-compatible format
        self.tk_image1 = ImageTk.PhotoImage(self.image1)
        self.tk_image1_copy = ImageTk.PhotoImage(self.image1_copy)
        self.tk_image2 = ImageTk.PhotoImage(self.image2)
        self.tk_image2_copy = ImageTk.PhotoImage(self.image2_copy)

        # Create labels for images and place them in grid
        tk.Label(self.root, image = self.tk_image1).grid(row = 0, column = 0)
        tk.Label(self.root, image = self.tk_image1_copy).grid(row = 0, column = 1)
        tk.Label(self.root, image = self.tk_image2).grid(row = 1, column = 0)
        tk.Label(self.root, image = self.tk_image2_copy).grid(row = 1, column = 1)

    def on_button_pressed(self):
        print("Button pressed")
        self.image1_copy = self.image2_copy
        self.update_canvas()


"""
import cv2
import numpy as np
from PIL import Image

# Open the image using PIL
img1 = Image.open('path_to_image.jpg')

# Convert the PIL image to a NumPy array
open_cv_image = np.array(img1)

# Convert RGB to BGR
open_cv_image = open_cv_image[:, :, ::-1]

gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)


# Convert BGR to RGB
rgb_image = gray_image[:, :, ::-1] if len(gray_image.shape) == 3 else gray_image

# Convert the NumPy array back to a PIL Image
pil_image = Image.fromarray(rgb_image)
"""