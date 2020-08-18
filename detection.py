from queue import Queue
import numpy as np

building_color = [255,255,255]  # class 1
tree_color = [0,255,0]      # class 5
illegal_color = [255, 0, 0]
Size=(800,800)

def Process(msk,i,j):
    global visit, building_color, tree_color
    que = Queue()
    count_building, count_green = 0, 0
    que.put((i,j))
    while not que.empty():
        i, j = que.get()
        if i not in range (0,Size[0]) or j not in range (0,Size[1]):
            continue
        if visit[i][j]:
            continue
        elif (msk[i][j] == tree_color).all():
            count_green += 1
        elif (msk[i][j] == building_color).all():
            count_building += 1
            que.put((i+1,j))    ### push all nighbours
            que.put((i-1,j))    ### push all nighbours
            que.put((i,j+1))    ### push all nighbours
            que.put((i,j-1))    ### push all nighbours
        visit[i][j] = 1
    return count_building, count_green


def Fill(i,j,res,msk):
    global illegal_color
    vis = np.zeros(Size).astype(np.uint8)
    color = msk[i][j]
    que = Queue()
    que.put((i,j))
    while not que.empty():
        i, j = que.get()
        if i not in range (0,Size[0]) or j not in range (0,Size[1]):
            continue
        if vis[i][j] == 1 or (msk[i][j] != color).any():
            continue
        else:
            res[i][j] = illegal_color # mark illegal building
            que.put((i+1,j))    ### push all nighbours
            que.put((i-1,j))    ### push all nighbours
            que.put((i,j+1))    ### push all nighbours
            que.put((i,j-1))    ### push all nighbours
        vis[i][j] = 1
    return res

def Detect(mask,coff=10):
    global visit, building_color,Size
    visit = np.zeros((mask.shape[0], mask.shape[1]))
    res = np.zeros(mask.shape)
    Size = (mask.shape[0], mask.shape[1])
    for i in range(mask.shape[0]):
        for j in range (mask.shape[1]):
            if (mask[i][j] == building_color).all() and visit[i][j] == 0:
                count_building, count_green = Process(msk=mask,i=i,j=j)
                if coff * count_green > count_building:
                    # this is illagl building
                    res = Fill(i=i,j=j,res=res,msk=mask)
    return res
