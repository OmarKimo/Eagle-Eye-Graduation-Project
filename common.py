
iconsPath = './icons/'
savedResultsPath = './results/'
testImagesPath = './testImages/'

def colorPred(num):
    assert num in range(0,11)

    # buildings & manmade structures
    if num < 2: return (255,255,255)
    # Road & Track
    if num < 4: return (255,255,0)
    # Trees & Crops
    if num < 6: return (0,255,0)
    # Waterway & Standing water
    if num < 8: return (0,0, 255)
    # Vehicle Large & Small
    if num < 10: return (0,255,255)
    # unspecified
    if num == 10: return (0,0,0)
