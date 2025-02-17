import numpy as np
import tkinter as tk
from tkinter import ttk
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

        # Add kernel size selection dropdown
        self.size_options = ["3 x 3", "5 x 5"]
        self.size_selector = ttk.Combobox(self.root, values=self.size_options, state="readonly")
        self.size_selector.set("Select a Kernel Size")
        self.size_selector.grid(row=3, column=0, pady=10)
        self.size_selector.bind("<<ComboboxSelected>>", self.on_kernel_selected)

        # Add filter selection dropdown
        self.filter_options = ["Box Filter", "Box Filter (OpenCV)", "X-axis Sobel Filter", "Y-axis Sobel Filter", "XY-axis Sobel Filter", "XY-axis Sobel Filter (OpenCV)", "Gaussian Filter (OpenCV)"]
        self.filter_selector = ttk.Combobox(self.root, values = self.filter_options, state = "readonly")
        self.filter_selector.set("Select a Filter")
        self.filter_selector.grid(row = 4, column = 0, pady = 10)
        self.filter_selector.bind("<<ComboboxSelected>>", self.on_filter_selected)

        # Add reset button
        self.button = tk.Button(self.root, text="Reset", command = self.on_reset_button_pressed)
        self.button.grid(row = 3, column = 1, pady = 10)

        # Start main event loop
        self.root.mainloop()

    def update_canvas(self):
        # Recompute Tkinter image format
        self.tk_image1 = ImageTk.PhotoImage(self.image1)
        self.tk_image1_copy = ImageTk.PhotoImage(self.image1_copy)
        self.tk_image2 = ImageTk.PhotoImage(self.image2)
        self.tk_image2_copy = ImageTk.PhotoImage(self.image2_copy)

        # Create labels for images and place them in grid
        tk.Label(self.root, image = self.tk_image1).grid(row = 0, column = 0)
        tk.Label(self.root, image = self.tk_image1_copy).grid(row = 0, column = 1)
        tk.Label(self.root, image = self.tk_image2).grid(row = 1, column = 0)
        tk.Label(self.root, image = self.tk_image2_copy).grid(row = 1, column = 1)



    def on_filter_selected(self, event):
        selected_filter = self.filter_selector.get()
        print(f"New filter selected: {selected_filter}\n")

        # Execute selection
        self.execute_selection()

        # Update canvas
        self.update_canvas()



    def on_kernel_selected(self, event):
        selected_kernel = self.size_selector.get()
        print(f"New kernel selected: {selected_kernel}\n")

        # Execute selection
        self.execute_selection()

        # Update canvas
        self.update_canvas()



    def on_reset_button_pressed(self):
        print("Reset button pressed!\n")

        print(f"{self.size_selector.get()} | {self.filter_selector.get()}\n")

        print("Resetting...\n")

        # Reset filter selection and size to selector to default state
        self.filter_selector.set("Select a Filter")
        self.size_selector.set("Select a Kernel Size")

        print(f"{self.size_selector.get()} | {self.filter_selector.get()}\n")

        # Update canvas
        self.update_canvas()



    def execute_selection(self):
        # Test to see if both selections are complete
        if self.filter_selector.get() == "Select a Filter" or self.size_selector.get() == "Select a Kernel Size":
            print("Incomplete selection\n")
            return

        # Execute corresponding filter function
        if self.filter_selector.get() == "Box Filter":
            self.box_filter()



    def box_filter(self):
        print("Beginning box filter conversion...\n")

        # Load images
        img1_cv = np.array(self.image1)
        img2_cv = np.array(self.image2)

        # Determine kernel size
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        if self.filter_selector.get() == "5 x 5":
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

        # Determine image dimensions
        height1, width1, channels1 = img1_cv.shape
        height2, width2, channels2 = img2_cv.shape

        # Pad images
        pad = kernel_size // 2
        padded_img1 = np.pad(img1_cv, ((pad, pad), (pad, pad), (0, 0)), mode = 'reflect')
        padded_img2 = np.pad(img2_cv, ((pad, pad), (pad, pad), (0, 0)), mode = 'reflect')

        # Create empty output canvases
        output1 = np.zeros_like(img1_cv, dtype = np.uint8)
        output2 = np.zeros_like(img2_cv, dtype = np.uint8)

        # Compute box filter for image 1
        for c1 in range(channels1):
            for h1 in range(height1):
                for w1 in range(width1):
                    # Load pixel values from region
                    pixels1 = padded_img1[h1 : h1 + kernel_size, w1 : w1 + kernel_size, c1]

                    # Execute computation
                    output1[h1, w1, c1] = np.clip(np.sum(pixels1 * kernel), 0, 255)

        # Compute box filter for image 2
        # for c2 in range(channels2):
        #     for h2 in range(height2):
        #         for w2 in range(width2):
        #             # Load pixel values from region
        #             pixels2 = padded_img2[h2: h2 + kernel_size, w2: w2 + kernel_size, c2]
        #
        #             # Execute computation
        #             output1[h2, w2, c2] = np.clip(np.sum(pixels2 * kernel), 0, 255)

        # Reformat new images
        new_image1 = Image.fromarray(output1)
        # new_image2 = Image.fromarray(output2)

        # Update global images
        self.image1_copy = new_image1
        # self.image2_copy = new_image2



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

# Add UI elements
        self.button = tk.Button(self.root, text="Press Me", command=self.on_button_pressed)
        self.button.grid(row=2, column=0, columnspan=2)
"""