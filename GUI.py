from tkinter import *
from tkinter.ttk import Combobox
from tkinter import filedialog
import os
import time
from PIL import ImageTk,Image
from common import iconsPath, testImagesPath


mp = {
        '3 bands': 3,
        '4 bands': 4,
        '20 bands': 20,
    }

def Welcome():
    global window
    window = Tk()
    window.title("Eagle Eye")
    Label(window, text = 'Welcome to Eagle Eye', font =('Verdana', 15)).pack(side = TOP, pady = 10)
    photo = PhotoImage(file = iconsPath+"icon.png")
    Button(window, image = photo, command=EndGUI,compound = LEFT).pack(side = TOP)
    mainloop()

def EndGUI():
    global window
    window.destroy()

def GUIInput():
    global file,choose,fileIN,window, mp, background_image
    window = Tk()
    window.title("Eagle Eye")
    window.geometry('800x600')  # setting window size (widthxhight)
    background_image = PhotoImage(file = iconsPath+"1.png")
    background_label = Label(window, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    load = Image.open(iconsPath+"icon.png")
    render = ImageTk.PhotoImage(load)
    img = Label(window, image=render)
    img.image = render
    img.place(x=0, y=500)
    i=0

    options = ['3 bands', '4 bands','20 bands']
    inputtype = Label(window, text="Choose the number of bands")
    #inputtype.pack(pady=20)
    inputtype.grid(row=i, column=0)
    choose = Combobox(window, state='readonly')
    choose['values'] = options
    choose.current(0)
    choose.grid(row=i, column=1)
    i=i+1

    fileLabel = Label(window, text="Enter image path")
    fileLabel.grid(row=i, column=0)
    fileIN = Entry(window)
    fileIN.grid(row=i, column=1)
    btn1 = Button(window, text="Browse", bg="white", fg="Black", command=GetFile, font=("Arial", 8))
    btn1.grid(row=i, column=2)
    i=i+1

    # Button
    btn3 = Button(window, text="Process", bg="white", fg="Black", command=TakeInput, font=("Arial", 20))
    btn3.grid(row=i, column=0)
    

    window.mainloop()


def GetFile():
    global window,file,fileIN
    file = filedialog.askopenfilename(initialdir=testImagesPath, title="Select image",filetypes=(("tif files", "*.tif"), ("all files", "*.*")))
    fileIN.delete(0, END)
    fileIN.insert(0, file)

def TakeInput():
    global bands,file,choose,fileIN,mp
    file = fileIN.get()
    bands= mp[choose.get()]
    EndGUI()

def GUIBuild():
    global bands,file
    Welcome()
    GUIInput()
    return bands,file

def ProgramTimer():
    global window,curruntstate,currunttime,hour,minute,second, background_image
    window = Tk()
    window.title("Eagle Eye")
    window.geometry('800x600')  # setting window size (widthxhight)
    background_image = PhotoImage(file=iconsPath+"1.png")
    background_label = Label(window, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    curruntstate = Label(window, text="Processing...",font=("Arial", 20))
    curruntstate.grid(row=3, column=0)

    currunttime = Label(window, text="",font=("Helvetica",70), bg="Green")
    currunttime.grid(row=10, column=0)
    Clock()
    return window
    #window.mainloop()


def UpdateText(newstring):
    global curruntstate
    curruntstate.config(text=newstring)


def Clock():
    global currunttime,hour,minute,second
    second=second+1
    if second==60:
        second=0
        minute=minute+1
    if minute==60:
        minute=0
        hour=hour+1
    mytext=""
    if hour < 10:
        mytext = mytext + "0"
    mytext = mytext + str(hour) + ":"

    if minute < 10:
        mytext = mytext + "0"
    mytext = mytext + str(minute) + ":"

    if second < 10:
        mytext = mytext + "0"
    mytext = mytext + str(second)

    currunttime.config(text=mytext)
    currunttime.after(1000,Clock)

bands, file = 3, ""
hour, minute, second = 0,0,0


