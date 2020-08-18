from GUI import GUIBuild, EndGUI, UpdateText, ProgramTimer
from ReadFile import ReadTif,ContrastAdjustment
from detection import Detect
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import threading
from UNetSegmentation import UNetSemanticSegmentation
from common import savedResultsPath

obj = None

#show the images
def showresult(image,result,final):
    fig = plt.figure()
    columns = 3
    rows = 1
    try:
        fig.add_subplot(rows, columns, 1)
        plt.title('Input Image')
        plt.imshow(image[:,:,0:3])
        #plt.show()
    except:
        print("Error in showing image ")
        return False

    try:
        fig.add_subplot(rows, columns, 2)
        plt.title('Segmented Image')
        plt.imshow(result)
        #plt.show()
    except:
        print("Error in showing result ")
        return False

    try:
        fig.add_subplot(rows, columns, 3)
        plt.title('Potential Violations')
        plt.imshow(final)
        #plt.show()
    except:
        print("Error in showing detection ")
        return False
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
    return True



class App(threading.Thread):
    def __init__(self, tk_root):
        self.root = tk_root
        threading.Thread.__init__(self)
        self.start()

    def run(self):
        global obj, file, bands
        UpdateText("Reading files...")
        time.sleep(1)
        img = ReadTif(file,bands)
        #UpdateText("Reading files is done ...")
        time.sleep(1)
        UpdateText("Processing...")
        time.sleep(1)

        # Omar
        img2 = ContrastAdjustment(img)
        _, colored_mask = obj.execute(img2)
        #print(colored_mask.shape)
        
            
        #UpdateText("Processing is done ...")
        time.sleep(1)
        UpdateText("Detecting Possible Violations")
        finalres=Detect(mask=colored_mask)
        time.sleep(1)

        UpdateText("Showing result...")
        showresult(img2[:,:,0:3], colored_mask, finalres)
        # if succ == True:
        #     UpdateText("Showing result is done ...")
        # else:
        #     UpdateText("Showing result is failed ...")
        time.sleep(1)
        UpdateText("Saving Result...")
        im = Image.fromarray(colored_mask.astype(np.uint8))
        saveName = savedResultsPath + file.split('/')[-1][:-4] + '_'
        im.save(saveName+'Segmented.tif')
        im = Image.fromarray(finalres.astype(np.uint8))
        im.save(saveName+'Violation.tif')

        time.sleep(1)
        UpdateText("Exiting ...")
        time.sleep(1)
        EndGUI()
        self.root.quit()
        self.root.update()
        


def Run():
    global window,bands,file, obj
    bands,file = GUIBuild()
    if file == "":
        return False
    obj = UNetSemanticSegmentation(bands)
    window=ProgramTimer()
    APP = App(window)
    window.mainloop()
    return True


Run()

print("Program End... bye bye")