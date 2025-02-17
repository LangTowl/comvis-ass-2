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
        self.image1 = ImageTk.PhotoImage(self.image1)
        self.image1_copy = ImageTk.PhotoImage(self.image1_copy)
        self.image2 = ImageTk.PhotoImage(self.image2)
        self.image2_copy = ImageTk.PhotoImage(self.image2_copy)

        # Render canvas
        self.update_canvas()

        # Start main event loop
        self.root.mainloop()

    def update_canvas(self):
        # Create labels for images and place them in grid
        tk.Label(self.root, image = self.image1).grid(row = 0, column = 0)
        tk.Label(self.root, image = self.image1_copy).grid(row = 0, column = 1)
        tk.Label(self.root, image = self.image2).grid(row = 1, column = 0)
        tk.Label(self.root, image = self.image2_copy).grid(row = 1, column = 1)
