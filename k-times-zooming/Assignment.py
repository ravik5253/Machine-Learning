# =============================================================================
# K-Times Zooming Algorithm
# input :- command line argument (path to input image , Pivot point , Scale)
# output :- zoomed image around pivot point with same shape as input image
# Issue :
# 1. The code handles only 3-channel images
# Solved: refactored it to work with C >= 1 input image. 
# =============================================================================
import cv2
import argparse
import numpy as np
import math

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="Path to input image", required=True)
ap.add_argument("-p", "--pivot-point", help="Pivot point coordinates x, y separated by comma (,)", required=True)
ap.add_argument("-s", "--scale", help="Scale to zoom", type=int, required=True)
args = vars(ap.parse_args())

image_path = args["image"]
x, y = map(int, args["pivot_point"].split(","))
scale = args["scale"]
image = cv2.imread(image_path)

#for grey scale 
channels = 1
if (len(image.shape)==3):
    channels = image.shape[2]
'''
Preprocessing steps:
Step 1 : Defining co-ordinates to create a window around the point of interest
Step 2 : The window is proportional to size of original image, to maintain aspect ratio
Step 3 : Defining the top left, and bottom right co-ordinates
Step 4 : The window is adjusted, if it happens to be going out of the scope of the original image size
Step 5 : The window is applied over the image to get only the required image portion  A - region of interest
'''

length = int((math.ceil)((image.shape[0]-1)*float(1)/float(scale)+1))
width = int((math.ceil)((image.shape[1]-1)*float(1)/float(scale)+1))

top_left_x = int(x - (float(1)/float(2) * width))
top_left_y = int(y- (float(1)/float(2) * length))

bottom_right_x = int(top_left_x + (width))
bottom_right_y = int(top_left_y + (length))


if (top_left_x < 0):
	bottom_right_x = bottom_right_x - top_left_x
	top_left_x = 0
if (top_left_y < 0):
	bottom_right_y = bottom_right_y - top_left_y
	top_left_y = 0

if (bottom_right_x > image.shape[1]):
	top_left_x = top_left_x - (bottom_right_x - image.shape[1])
	bottom_right_x = image.shape[1]
if (bottom_right_y > image.shape[0]):
	top_left_y = top_left_y - (bottom_right_y - image.shape[0])
	bottom_right_y = image.shape[0]

img = image.tolist()

A = [ [ img[i][j] for j in  range(top_left_x,bottom_right_x) ] for i in range(top_left_y,bottom_right_y) ]

"""
K-Times Zooming:
Step 1 : m = number of rows of the input matrix A. 
Step 2 : n = number of columns of the input matrix A.
Step 3 : Now the input matrix A with m rows , n columns and c channels , [A (m,n,c)] is created.
Step 4 : Now the elements of the matrix A are taken as input and the Matrix [A (m,n)] is prepared.
Step 5 : k = input for the amount of zooming required is taken as input.
Step 6: We now prepare a matrix B with m rows and {n + (n-­1)*(k-­1)} columns . [ Since for row wise zooming we need to insert (k­1) elements between each adjacent pixels of each row .].
Thus the B matrix now prepared is like ,
[ B (m,{n + (n-­1)*(k-­1)}) ]. Also, let us take , z=n+{(n-­1)*(k-­1)} or, z=(n-­1)*k+1; Step 7 : Now, the matrix B is prepared with the help of the following pseudo code: 
Step 7 : Now, the matrix B is prepared with the help of the following pseudo code: 
m=11,n=43
    """
m = len(A)
n = len(A[0])
k = scale
if (channels>1):
    B = [[[ 0 for x in range(channels)] for y in range((n-1)*k+1)] for z in range(m)]
    for i in range(m):
        j=0
        B[i][j] = [A[i][j][c] for c in range(channels)]
        for j in range(1,n):
            B[i][(j*k)] = [A[i][j][c] for c in range(channels)]
            S = [ ((B[i][j*k][c] -B[i][(j-1)*k][c])/k) for c in range(channels)]
            for l in range(k-1,0,-1): 
                B[i][(j*k)-l] = [np.round(B[i][(j-1)*k][c]+(k-l)*S[c]) for c in range(channels)]
else:
    B = [[0 for y in range((n-1)*k+1)] for z in range(m)]
    for i in range(m):
        j=0
        B[i][j] = A[i][j]
        for j in range(1,n):
            B[i][(j*k)] = A[i][j]
            S = (B[i][j*k] -B[i][(j-1)*k])/k
            for l in range(k-1,0,-1): 
                B[i][(j*k)-l] = np.round(B[i][(j-1)*k]+(k-l)*S)

    
"""
Step 8 : So, in the 7th step k times zooming is done row wise and an intermediate matrix B is obtained , 
Step 9 : Now after this we prepare a matrix C , which will be our output matrix , i.e. the matrix which is zoomed k times, column and row wise.
So, this matrix will be having { m + ((m-­1)*(k-­1))}rows and { n + ((n-­1)*(k-­1))} columns.
So, now the C matrix to be prepared will be like : [ C ({m + (m-­1)*(k-­1)},z) ].
Now lety=m+{(m­-1)*(k-­1)} or, y=(m-­1)*k+1;
Thus the C matrix prepared is like [C(y,z)].
Step 10 : Now the values in the C matrix are inserted with the following pseudo code in the following manner:
Step 11 : The output , that is the zoomed image matrix is the C matrix.
Step 12 :End
"""

if (channels>1):
    C = [[[ 0 for x in range(channels)] for y in range((n-1)*k+1)] for z in range((m-1)*k+1)]
    for i in range((n-1)*k+1):
        j=0
        C[j][i] = [ B[j][i][c] for c in range(channels) ]    
        for j in range(1,m):
            C[(j*k)][i] = [ B[j][i][c] for c in range(channels)]
            S = [ ( (C[j*k][i][c] - C[(j-1)*k][i][c])/k ) for c in range(channels)]
            for l in range(k-1,0,-1): 
                C[(j*k)-l][i] = [np.round(C[(j-1)*k][i][c]+(k-l)*S[c]) for c in range(channels)]        
else:
    C = [[0 for y in range((n-1)*k+1)] for z in range((m-1)*k+1)]
    for i in range((n-1)*k+1):
        j=0
        C[j][i] = B[j][i]
        for j in range(1,m):
            C[(j*k)][i] = B[j][i]
            S = (C[j*k][i] - C[(j-1)*k][i])/k 
            for l in range(k-1,0,-1): 
                C[i][(j*k)-l] = np.round(C[(j-1)*k][i]+(k-l)*S)

'''
Assumption 
Crop image if it is large than actual
Step 1 : Find Difference between actual and zoomed image shape
Step 2 : Slice list from left, right, bottom, top cropped to same
'''
original_width = image.shape[1] 
original_height = image.shape[0] 
new_width = (n-1)*scale+1
new_height = (m-1)*scale+1
if (original_height != new_height):
    diff = new_height-original_height
    st = int(diff/2)
    en = int(new_height-diff+st)
    C = C[st:en]

if (original_width != new_width):
    diff = new_width-original_width
    st = int(diff/2)
    en = int(new_width-diff+st)
    C = [C[i][st:en] for i in range(original_height)]
    
cv2.imwrite("zoomed_image.png", np.array(C, dtype="uint8"))





