import cv2
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
        self.size_selector = ttk.Combobox(self.root, values = self.size_options, state = "readonly")
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
        self.image1_copy = self.image1
        self.image2_copy = self.image2

        print(f"{self.size_selector.get()} | {self.filter_selector.get()}\n")

        # Update canvas
        self.update_canvas()



    def execute_selection(self):
        # Test to see if all required fields are filled
        if self.filter_selector.get() == "Select a Filter" or self.size_selector.get() == "Select a Kernel Size":
            print("Incomplete selection\n")
            return

        # Execute corresponding filter function
        if self.filter_selector.get() == "Box Filter":
            self.box_filter()
        elif self.filter_selector.get() == "Box Filter (OpenCV)":
            self.box_filter_opencv()
        elif self.filter_selector.get() == "X-axis Sobel Filter":
            self.sobel_filter("x")
        elif self.filter_selector.get() == "Y-axis Sobel Filter":
            self.sobel_filter("y")
        elif self.filter_selector.get() == "XY-axis Sobel Filter":
            self.sobel_filter("xy")
        elif self.filter_selector.get() == "XY-axis Sobel Filter (OpenCV)":
            self.sobel_filter_cv()
        elif self.filter_selector.get() == "Gaussian Filter (OpenCV)":
            self.gaussian_filter()



    def box_filter(self):
        print("Beginning filter conversion...\n")

        # Load images
        img1_cv = np.array(self.image1)
        img2_cv = np.array(self.image2)

        # Determine kernel size
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        if self.size_selector.get() == "5 x 5":
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
        for c2 in range(channels2):
            for h2 in range(height2):
                for w2 in range(width2):
                    # Load pixel values from region
                    pixels2 = padded_img2[h2: h2 + kernel_size, w2 : w2 + kernel_size, c2]

                    # Execute computation
                    output2[h2, w2, c2] = np.clip(np.sum(pixels2 * kernel), 0, 255)

        # Reformat new images
        new_image1 = Image.fromarray(output1)
        new_image2 = Image.fromarray(output2)

        # Update global images
        self.image1_copy = new_image1
        self.image2_copy = new_image2

        print("Conversion complete.\n")



    def box_filter_opencv(self):
        print("Beginning filter conversion...\n")

        # Convert images to compatible format
        img1_cv = cv2.cvtColor(np.array(self.image1), cv2.COLOR_RGB2BGR)
        img2_cv = cv2.cvtColor(np.array(self.image2), cv2.COLOR_RGB2BGR)

        # Determine which kernel size to use
        ksize = (3, 3) if self.size_selector.get() == "3 x 3" else (5, 5)

        # Execute modification
        temp1 = cv2.boxFilter(img1_cv, ddepth = -1, ksize = ksize)
        temp2 = cv2.boxFilter(img2_cv, ddepth = -1, ksize = ksize)

        # BGR -> RGB
        temp1 = cv2.cvtColor(temp1, cv2.COLOR_RGB2BGR)
        temp2 = cv2.cvtColor(temp2, cv2.COLOR_RGB2BGR)

        # Convert back to PIL image
        self.image1_copy = Image.fromarray(temp1)
        self.image2_copy = Image.fromarray(temp2)

        print("Conversion complete.\n")



    def sobel_filter(self, axis = str):
            print("Beginning filter conversion...\n")

            # Load images
            img1_cv = cv2.cvtColor(np.array(self.image1), cv2.COLOR_RGB2BGR)
            img2_cv = cv2.cvtColor(np.array(self.image2), cv2.COLOR_RGB2BGR)

            # Select kernel
            kernel_x = None
            kernel_y = None
            filter_size = 3 if self.size_selector.get() == "3 x 3" else 5
            offset = filter_size // 2

            if axis == "x" or axis == "xy":
                kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] if self.size_selector.get() == "3 x 3" else [[-2, -1, 0, 1, 2], [-3, -2, 0, 2, 3], [-4, -3, 0, 3, 4], [-3, -2, 0, 2, 3], [-2, -1, 0, 1, 2]]

            if axis == "y" or axis == "xy":
                kernel_y = [[-1, 2, -1], [0, 0, 0], [1, 2, 1]] if self.size_selector.get() == "3 x 3" else [[-2, -3, -4, -3, -2], [-1, -2, -3, -2, -1], [0,  0,  0,  0,  0], [1,  2,  3,  2,  1], [2,  3,  4,  3,  2]]

            # Determine image dimensions
            height1, width1, channels1 = img1_cv.shape
            height2, width2, channels2 = img2_cv.shape

            # Create empty output canvases
            output1 = np.zeros_like(img1_cv, dtype = np.uint8)
            output2 = np.zeros_like(img2_cv, dtype = np.uint8)

            if axis == "xy":
                gradient_x1 = np.zeros_like(img1_cv, dtype = np.uint8)
                gradient_y1 = np.zeros_like(img1_cv, dtype = np.uint8)
                gradient_x2 = np.zeros_like(img2_cv, dtype = np.uint8)
                gradient_y2 = np.zeros_like(img2_cv, dtype = np.uint8)

            # Apply filter to image 1
            for h1 in range(offset, height1 - offset):
                for w1 in range(offset, width1 - offset):
                    # Pull channels pixel values for convolution
                    red1 = img1_cv[h1 - offset: h1 + offset + 1, w1 - offset: w1 + offset + 1, 2]
                    green1 = img1_cv[h1 - offset: h1 + offset + 1, w1 - offset: w1 + offset + 1, 1]
                    blue1 = img1_cv[h1 - offset: h1 + offset + 1, w1 - offset: w1 + offset + 1, 0]

                    # Convolve
                    if axis == "x" or axis == "y":
                        output1[h1, w1, 2] = np.sum(red1 * (kernel_x if axis == "x" else kernel_y))
                        output1[h1, w1, 1] = np.sum(green1 * (kernel_x if axis == "x" else kernel_y))
                        output1[h1, w1, 0] = np.sum(blue1 * (kernel_x if axis == "x" else kernel_y))

                    if axis == "xy":
                        gradient_x1[h1, w1, 2] = np.sum(red1 * kernel_x)
                        gradient_x1[h1, w1, 1] = np.sum(green1 * kernel_x)
                        gradient_x1[h1, w1, 0] = np.sum(blue1 * kernel_x)

                        gradient_y1[h1, w1, 2] = np.sum(red1 * kernel_y)
                        gradient_y1[h1, w1, 1] = np.sum(green1 * kernel_y)
                        gradient_y1[h1, w1, 0] = np.sum(blue1 * kernel_y)

            # Apply filter to image 2
            for h2 in range(offset, height2 - offset):
                for w2 in range(offset, width2 - offset):
                    # Pull channels pixel values for convolution
                    red2 = img2_cv[h2 - offset: h2 + offset + 1, w2 - offset: w2 + offset + 1, 2]
                    green2 = img2_cv[h2 - offset: h2 + offset + 1, w2 - offset: w2 + offset + 1, 1]
                    blue2 = img2_cv[h2 - offset: h2 + offset + 1, w2 - offset: w2 + offset + 1, 0]

                    # Convolve
                    if axis == "x" or axis == "y":
                        output2[h2, w2, 2] = np.sum(red2 * (kernel_x if axis == "x" else kernel_y))
                        output2[h2, w2, 1] = np.sum(green2 * (kernel_x if axis == "x" else kernel_y))
                        output2[h2, w2, 0] = np.sum(blue2 * (kernel_x if axis == "x" else kernel_y))

                    if axis == "xy":
                        gradient_x2[h2, w2, 2] = np.sum(red2 * kernel_x)
                        gradient_x2[h2, w2, 1] = np.sum(green2 * kernel_x)
                        gradient_x2[h2, w2, 0] = np.sum(blue2 * kernel_x)

                        gradient_y2[h2, w2, 2] = np.sum(red2 * kernel_y)
                        gradient_y2[h2, w2, 1] = np.sum(green2 * kernel_y)
                        gradient_y2[h2, w2, 0] = np.sum(blue2 * kernel_y)

            if axis == "x" or axis == "y":
                new_image1 = Image.fromarray(output1)
                new_image2 = Image.fromarray(output2)

                self.image1_copy = new_image1
                self.image2_copy = new_image2

            if axis == "xy":
                mag1 = np.sqrt(gradient_x1 ** 2 + gradient_y1 ** 2)
                mag1 = np.uint8((mag1 / np.max(mag1)) * 255)
                output_xy1 = Image.fromarray(np.uint8(mag1))
                self.image1_copy = output_xy1

                mag2 = np.sqrt(gradient_x2 ** 2 + gradient_y2 ** 2)
                mag2 = np.uint8((mag2 / np.max(mag2)) * 255)
                output_xy2 = Image.fromarray(np.uint8(mag2))
                self.image2_copy = output_xy2

            print("Conversion complete.\n")



    def sobel_filter_cv(self):
        print("Beginning filter conversion...\n")

        # Convert images to compatible format
        img1_cv = cv2.cvtColor(np.array(self.image1), cv2.COLOR_RGB2BGR)
        img2_cv = cv2.cvtColor(np.array(self.image2), cv2.COLOR_RGB2BGR)

        # Determine dimension
        dimension = 3 if self.size_selector.get() == "3 x 3" else 5

        # Manipulate images
        new_img1 = cv2.Sobel(img1_cv, cv2.CV_64F, dx = 0, dy = 1, ksize = dimension)
        new_img2 = cv2.Sobel(img2_cv, cv2.CV_64F, dx = 0, dy = 1, ksize = dimension)

        # Convert back
        self.image1_copy = Image.fromarray(np.uint8(new_img1))
        self.image2_copy = Image.fromarray(np.uint8(new_img2))

        print("Conversion complete.\n")



    def gaussian_filter(self):
        print("Beginning filter conversion...\n")

        # Convert images to compatible format
        img1_cv = cv2.cvtColor(np.array(self.image1), cv2.COLOR_RGB2BGR)
        img2_cv = cv2.cvtColor(np.array(self.image2), cv2.COLOR_RGB2BGR)

        # Determine dimension
        dimension = (3, 3) if self.size_selector.get() == "3 x 3" else (5, 5)

        # Manipulate images
        new_img1 = cv2.GaussianBlur(img1_cv, dimension, 0)
        new_img2 = cv2.GaussianBlur(img2_cv, dimension, 0)

        # Convert back
        new_img1 = cv2.cvtColor(new_img1, cv2.COLOR_BGR2RGB)
        new_img2 = cv2.cvtColor(new_img2, cv2.COLOR_BGR2RGB)
        self.image1_copy = Image.fromarray(np.uint8(new_img1))
        self.image2_copy = Image.fromarray(np.uint8(new_img2))

        print("Conversion complete.\n")
