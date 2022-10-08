from tkinter import *
from PIL import ImageTk, Image, ImageFilter
from tkinter import filedialog
import numpy as np
import cv2
import PIL.Image, PIL.ImageTk
import tkinter.font as font
import cv2 as cv
import numpy as np

root = Tk()
root.title("Image Processing")
root.geometry("790x680+280+20")
frame_tool = Frame(root, bg='#C8BBBE', height=790)
frame_tool.pack(fill='x')
myFont = font.Font(family="Times New Roman", size=14, weight="bold")


def identify_image():  # select image
    global canvas, IMG, image, image_tk
    canvas = Canvas(root, width=390, height=370)
    canvas.place(x=200, y=150)
    IMG = filedialog.askopenfilename()
    image = Image.open(IMG)
    image_tk = ImageTk.PhotoImage(image)
    image = image.resize((400, 380), Image.ANTIALIAS)
    image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=NW, image=image_tk)


def Enchanc():  # improve contrast & quality of image
    img = cv2.imread(IMG, 0)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def maxFilter():  # remove negative noise
    img = cv2.imread(IMG, 0)
    img = Image.fromarray((img))
    img = img.filter(ImageFilter.MaxFilter())
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def minFilter():  # remove positive noise
    img = cv2.imread(IMG, 0)
    img = Image.fromarray((img))
    img = img.filter(ImageFilter.MinFilter())
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def Remove():  # remove noise
    img = cv2.imread(IMG, 0)
    kern = np.array([[1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9]])
    img = cv2.filter2D(img, -1, kern)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def medianFilter():  # remove salt&papper noise
    img = cv2.imread(IMG, 0)
    img = Image.fromarray((img))
    img = img.filter(ImageFilter.MedianFilter())
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def bluring():  # reduce the edge content and makes transition very smooth
    img = cv2.imread(IMG, 0)
    img = Image.fromarray((img))
    img = img.filter(ImageFilter.BLUR())
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def lab():  # find areas of rapid change (edges) in images. Since derivative filters are very sensitive to noise, it is common to smooth the image
    img = cv2.imread(IMG, 0)
    img = cv2.Laplacian(img, -1, ksize=29, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def sobel():
    img = cv2.imread(IMG, 0)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def log():  # Enhance details in the darker regions of an image
    img = cv2.imread(IMG, 0)
    img = np.uint8(np.log1p(img))
    thresh = 3
    img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def negative():  # inverts image color/pixel's value
    img = cv2.imread(IMG, 0)
    img = 255 - img
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def increaseContrast():  # increase contrast
    img = cv2.imread(IMG, 0)
    alpha = 2
    beta = 50
    img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def conDecrease():  # Decrease contrast
    img = cv2.imread(IMG, 0)
    kern = np.array([[-1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1],
                     [-1, -1, 25, -1, -1],
                     [-1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1]])
    img = cv2.filter2D(img, -1, kern)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def gassian():  # reduce image noise and reduce detail "low light"
    img = cv2.imread(IMG, 0)
    img = cv2.GaussianBlur(img, (37, 37), 0)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


#def Alpha(): 
  #  x, y = np.shape(image) 
  #  m, n = np.shape(mask)
   # new_image = np.ones((x + m - 1, y + n - 1), np.uint8) 
    #fin_image = np.ones((x, y), np.uint8)  
    #m = m // 2
    #n = n // 2
   # new_image[m:new_image.shape[0] - m, n:new_image.shape[1] - n] = image  
   # return fin_image


def Geometric():  # computing an approximation of the gradient of the image intensity function
    img = cv2.imread(IMG, 0)
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(img, -1, kernelx)
    img_prewitty = cv2.filter2D(img, -1, kernely)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def fourier():
    img = cv2.imread(IMG, 0)
    img = np.fft.fft2(img)
    shift = np.fft.fftshift(img)
    img = 20 * np.log(np.abs(shift))
    img = np.uint8(img)
    img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


B1 = Button(root, text="Exit", font=myFont, fg='#7E354D', bg='white', command=root.destroy)
B1.place(x=700, y=620, height=50, width=70)

B3 = Button(root, text="Enhancement", font=myFont, fg='#7E354D', bg='#F9B7FF', command=Enchanc)
B3.place(x=50, y=150, height=50, width=120)

B4 = Button(root, text="Max", font=myFont, fg='#7E354D', bg='#F9B7FF', command=maxFilter)
B4.place(x=465, y=70, height=50, width=120)

B5 = Button(root, text="Min", font=myFont, fg='#7E354D', bg='#F9B7FF', command=minFilter)
B5.place(x=215, y=70, height=50, width=120)

B5 = Button(root, text="Harmonic", font=myFont, fg='#7E354D', bg='#F9B7FF', command=Remove)
B5.place(x=340, y=70, height=50, width=120)

B6 = Button(root, text="median", font=myFont, fg='#7E354D', bg='#F9B7FF', command=medianFilter)
B6.place(x=50, y=490, height=50, width=120)

B7 = Button(root, text="Arithmetic", font=myFont, fg='#7E354D', bg='#F9B7FF', command=bluring)
B7.place(x=630, y=420, height=50, width=120)

#B8 = Button(root, text="Laplace", font=myFont, fg='#7E354D', bg='#F9B7FF', command=lab)
#B8.place(x=50, y=360, height=50, width=120)

B9 = Button(root, text="sobel", font=myFont, fg='#7E354D', bg='#F9B7FF', command=sobel)
B9.place(x=630, y=295, height=50, width=120)

B10 = Button(root, text="thresholding", fg='#7E354D', bg='#F9B7FF', font=myFont, command=log)
B10.place(x=630, y=360, height=50, width=120)

B11 = Button(root, text="Negative", fg='#7E354D', bg='#F9B7FF', font=myFont, command=negative)
B11.place(x=630, y=150, height=50, width=120)

B12 = Button(root, text="cont_H Pos", fg='#7E354D', bg='#F9B7FF', font=myFont, command=increaseContrast)
B12.place(x=280, y=560, height=50, width=120)

B14 = Button(root, text="Gaussion", fg='#7E354D', bg='#F9B7FF', font=myFont, command=gassian)
B14.place(x=410, y=560, height=50, width=120)

B13 = Button(root, text="cont_H Neg", fg='#7E354D', bg='#F9B7FF', font=myFont, command=conDecrease)
B13.place(x=50, y=225, height=50, width=120)

#B15 = Button(root, text="Alpha", fg='#7E354D', bg='#F9B7FF', font=myFont, command=Alpha)
#B15.place(x=630, y=225, height=50, width=120)

B16 = Button(root, text="Geometric", fg='#7E354D', bg='#F9B7FF', font=myFont, command=Geometric)
B16.place(x=50, y=295, height=50, width=120)

B17 = Button(root, text=" midpoint", fg='#7E354D', bg='#F9B7FF', font=myFont, command= fourier)
B17.place(x=630, y=490, height=50, width=120)

B18 = Button(root, text="Fourier", fg='#7E354D', bg='#F9B7FF', font=myFont, command=fourier)
B18.place(x=50, y=420, height=50, width=120)

B2 = Button(root, text="Choose Image", fg='#7E354D', bg='white', font=myFont, command=identify_image,
            activebackground="#F9B7FF")
B2.place(x=530, y=620, height=50, width=160)

root.mainloop()
