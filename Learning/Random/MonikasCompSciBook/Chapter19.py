from swampy.Gui import *
import tkinter as tk

# Exercise 19.1 - Button in a button
g = Gui()

g.title('Button in a Button')
def make_label():
    g.la('Nice job!')

def make_button():
    g.bu(text='Again!', command=make_label)

g.bu(text='Press me', command=make_button)

g.mainloop()

# 19.2 - Using Canvas
def draw_circle():
    canvas.create_oval(50, 50, 150, 150, outline='black', fill='blue', width=2)

root = tk.Tk()
root.title('Canvas Circle Drawer')
canvas = tk.Canvas(root, width=200, height=200)
canvas.pack()

button = tk.Button(root, text='Draw Circle', command=draw_circle)
button.pack()

root.mainloop()
