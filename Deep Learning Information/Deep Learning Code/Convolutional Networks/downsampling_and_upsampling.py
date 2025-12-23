import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray

# Define 4 by 4 array for downsampling functions
orig_4_4 = np.array([[1, 3, 5,3 ], [6,2,0,8], [4,6,1,4], [2,8,0,3]])

# Define a subsampling function, which takes the top-left pixel of each 2 by 2 patch
def subsample(x_in):
  x_out = np.zeros(( int(np.ceil(x_in.shape[0]/2)), int(np.ceil(x_in.shape[1]/2)) ))
  kernel_height = 2
  kernel_width = 2
  height = int(np.ceil(x_in.shape[0]/2))
  width =  int(np.ceil(x_in.shape[1]/2))
  for h in range(height): 
    for w in range(width):
        h_start = h * kernel_height
        h_end   = min(h_start + kernel_height, x_in.shape[0])
        w_start = w * kernel_width
        w_end   = min(w_start + kernel_width, x_in.shape[1])
        patch = x_in[h_start:h_end, w_start:w_end]
        x_out[h, w] = patch[0, 0]

  return x_out

# Print the original 4 by 4 array
print("Original:")
print(orig_4_4)

# Print the subsampled array (2 by 2)
print("Subsampled:")
print(subsample(orig_4_4))

# Import the image
image = Image.open('test_image.png')

# Convert the imported image into a Numpy array
data = asarray(image)
data_subsample = subsample(data)

# Draw the original image 
plt.figure(figsize=(5,5))
plt.imshow(data, cmap='gray')
plt.show()

# Draw the subsampled image
plt.figure(figsize=(5,5))
plt.imshow(data_subsample, cmap='gray')
plt.show()

# Draw the subsampled image of the subsampled image 
data_subsample2 = subsample(data_subsample)
plt.figure(figsize=(5,5))
plt.imshow(data_subsample2, cmap='gray')
plt.show()

# Draw the subsampled image of the subsmapled image of the subsampled image 
data_subsample3 = subsample(data_subsample2)
plt.figure(figsize=(5,5))
plt.imshow(data_subsample3, cmap='gray')
plt.show()

# Define the maxpooling function, which keeps the maximum value of each 2 by 2 patch
def maxpool(x_in):
  kernel_height = 2
  kernel_width = 2
  height = int(np.ceil(x_in.shape[0]/2))
  width =  int(np.ceil(x_in.shape[1]/2))
  x_out = np.zeros((height, width))
  for h in range(height): 
    for w in range(width):
        h_start = h * kernel_height
        h_end   = min(h_start + kernel_height, x_in.shape[0])
        w_start = w * kernel_width
        w_end   = min(w_start + kernel_width, x_in.shape[1])
        patch = x_in[h_start:h_end, w_start:w_end]
        x_out[h, w] = np.max(patch)

  return x_out

# Print the original 4 by 4 array
print("Original:")
print(orig_4_4)

# Print the maxpooled array (2 by 2)
print("Maxpooled:")
print(maxpool(orig_4_4))

# Draw the original image 
plt.figure(figsize=(5,5))
plt.imshow(data, cmap='gray')
plt.show()

# Draw the maxpooled image 
data_maxpool = maxpool(data)
plt.figure(figsize=(5,5))
plt.imshow(data_maxpool, cmap='gray')
plt.show()

# Draw the maxpooled image of the maxpooled image 
data_maxpool2 = maxpool(data_maxpool)
plt.figure(figsize=(5,5))
plt.imshow(data_maxpool2, cmap='gray')
plt.show()

# Draw the maxpooled image of the maxpooled image of the maxpooled image 
data_maxpool3 = maxpool(data_maxpool2)
plt.figure(figsize=(5,5))
plt.imshow(data_maxpool3, cmap='gray')
plt.show()

# Define the meanpooling function, which takes the average value of each 2 by 2 patch
def meanpool(x_in):
  kernel_height = 2
  kernel_width = 2
  height = int(np.ceil(x_in.shape[0]/2))
  width =  int(np.ceil(x_in.shape[1]/2))
  x_out = np.zeros((height, width))
  for h in range(height): 
    for w in range(width):
        h_start = h * kernel_height
        h_end   = min(h_start + kernel_height, x_in.shape[0])
        w_start = w * kernel_width
        w_end   = min(w_start + kernel_width, x_in.shape[1])
        patch = x_in[h_start:h_end, w_start:w_end]
        x_out[h, w] = np.mean(patch)
  return x_out

# Print the original 4 by 4 array
print("Original:")
print(orig_4_4)

# Print the meanpooled array (2 by 2)
print("Meanpooled:")
print(meanpool(orig_4_4))

# Draw the original image
plt.figure(figsize=(5,5))
plt.imshow(data, cmap='gray')
plt.show()

# Draw the meanpooled image
data_meanpool = meanpool(data)
plt.figure(figsize=(5,5))
plt.imshow(data_meanpool, cmap='gray')
plt.show()

# Draw the meanpooled image of the meanpooled image
data_meanpool2 = meanpool(data_maxpool)
plt.figure(figsize=(5,5))
plt.imshow(data_meanpool2, cmap='gray')
plt.show()

# Draw the meanpooled image of the meanpooled image of the meanpooled image
data_meanpool3 = meanpool(data_meanpool2)
plt.figure(figsize=(5,5))
plt.imshow(data_meanpool3, cmap='gray')
plt.show()

# Define 2 by 2 array for the upsampling functions
orig_2_2 = np.array([[6, 8], [8,4]])

# Define the duplication upsampling function, which duplicates each pixel into a 2 by 2 patch
def duplicate(x_in):
  x_out = np.zeros(( x_in.shape[0]*2, x_in.shape[1]*2 ))
  kernel_height = 2
  kernel_width = 2
  height = int(np.ceil(x_in.shape[0]))
  width =  int(np.ceil(x_in.shape[1]))
  for h in range(height): 
    for w in range(width):
        h_start = h * kernel_height
        h_end   = h_start + kernel_height
        w_start = w * kernel_width
        w_end   = w_start + kernel_width
        x_out[h_start:h_end, w_start:w_end] = x_in[h, w]

  return x_out

# Print the original 2 by 2 array    
print("Original:")
print(orig_2_2)

# Print the duplicated array (4 by 4)
print("Duplicated:")
print(duplicate(orig_2_2))

# Draw the original image
plt.figure(figsize=(5,5))
plt.imshow(data_subsample3, cmap='gray')
plt.show()

# Draw the duplicated image
data_duplicate = duplicate(data_subsample3)
plt.figure(figsize=(5,5))
plt.imshow(data_duplicate, cmap='gray')
plt.show()

# Draw the duplicated image of the duplicated image
data_duplicate2 = duplicate(data_duplicate)
plt.figure(figsize=(5,5))
plt.imshow(data_duplicate2, cmap='gray')
plt.show()

# Draw the duplicated image of the duplicatee image of the duplicated image
data_duplicate3 = duplicate(data_duplicate2)
plt.figure(figsize=(5,5))
plt.imshow(data_duplicate3, cmap='gray')
plt.show()

# Define the max unpooling function, which takes a value from the original array and places it at the max location in each 2 by 2 patch
def max_unpool(x_in, x_high_res):
    kernel_height = 2
    kernel_width = 2
    height = x_in.shape[0]
    width  = x_in.shape[1]
    
    x_out = np.zeros((height*2, width*2))
    
    for h in range(height):
        for w in range(width):
            # Coordinates in high-res image
            h_start = h * kernel_height
            h_end   = min(h_start + kernel_height, x_high_res.shape[0])
            w_start = w * kernel_width
            w_end   = min(w_start + kernel_width, x_high_res.shape[1])
            
            # If the patch is empty, place a value at top-left of block
            if h_start >= h_end or w_start >= w_end:
                if h_start < x_out.shape[0] and w_start < x_out.shape[1]:
                    x_out[h_start, w_start] = x_in[h, w]
                continue

            patch = x_high_res[h_start:h_end, w_start:w_end]
            if patch.size == 0:
                # Place pooled value at top-left if patch is unexpectedly empty
                if h_start < x_out.shape[0] and w_start < x_out.shape[1]:
                    x_out[h_start, w_start] = x_in[h, w]
                continue

            # Find the index of the max in this patch
            max_idx = np.unravel_index(np.argmax(patch), patch.shape)
            
            # Place the pooled value at the max location
            out_h = h_start + max_idx[0]
            out_w = w_start + max_idx[1]
            if out_h < x_out.shape[0] and out_w < x_out.shape[1]:
                x_out[out_h, out_w] = x_in[h, w]
    
    return x_out

# Print the original 2 by 2 array
print("Original:")
print(orig_2_2)

# Print the max unpooled array (4 by 4)
print("Max unpooled:")
print(max_unpool(orig_2_2,orig_4_4))

# Draw the original image
plt.figure(figsize=(5,5))
plt.imshow(data_maxpool3, cmap='gray')
plt.show()

# Draw the max unpooled image
data_max_unpool= max_unpool(data_maxpool3,data_maxpool2)
plt.figure(figsize=(5,5))
plt.imshow(data_max_unpool, cmap='gray')
plt.show()

# Draw the max unpooled image of the max unpooled image
data_max_unpool2 = max_unpool(data_max_unpool, data_maxpool)
plt.figure(figsize=(5,5))
plt.imshow(data_max_unpool2, cmap='gray')
plt.show()

# Draw the max unpooled image of the max unpooled image of the max unpooled image
data_max_unpool3 = max_unpool(data_max_unpool2, data)
plt.figure(figsize=(5,5))
plt.imshow(data_max_unpool3, cmap='gray')
plt.show()

# Define the bilinear upsampling function, which uses bilinear interpolation to upsample an array/image
def bilinear(x_in):
  H, W = x_in.shape
  H_out, W_out = H * 2, W * 2
  x_out = np.zeros((H_out, W_out), dtype=float)
  for i in range(H_out):
    for j in range(W_out):
      x = i / 2.0
      y = j / 2.0
      x0 = int(np.floor(x)); y0 = int(np.floor(y))
      x1 = min(x0 + 1, H - 1); y1 = min(y0 + 1, W - 1)
      dx = x - x0; dy = y - y0
      if H == 1 and W == 1:
        x_out[i, j] = x_in[0, 0]
      elif H == 1:
        x_out[i, j] = (1 - dy) * x_in[0, y0] + dy * x_in[0, y1]
      elif W == 1:
        x_out[i, j] = (1 - dx) * x_in[x0, 0] + dx * x_in[x1, 0]
      else:
        x_out[i, j] = ((1 - dx) * (1 - dy) * x_in[x0, y0] +
                 dx * (1 - dy) * x_in[x1, y0] +
                 (1 - dx) * dy * x_in[x0, y1] +
                 dx * dy * x_in[x1, y1])
  return x_out.astype(x_in.dtype)  

# Print the original 2 by 2 array
print("Original:")
print(orig_2_2)

# Print the bilinearly upsampled array (4 by 4)
print("Bilinear:")
print(bilinear(orig_2_2))

# Draw the original image
plt.figure(figsize=(5,5))
plt.imshow(data_meanpool3, cmap='gray')
plt.show()

# Draw the bilinearly upsampled image
data_bilinear = bilinear(data_meanpool3)
plt.figure(figsize=(5,5))
plt.imshow(data_bilinear, cmap='gray')
plt.show()

# Draw the bilinearly upsampled image of the bilinearly upsampled image
data_bilinear2 = bilinear(data_bilinear)
plt.figure(figsize=(5,5))
plt.imshow(data_bilinear2, cmap='gray')
plt.show()

# Draw the bilinearly upsampled image of the bilinearly upsampled image of the bilinearly upsampled image
data_bilinear3 = duplicate(data_bilinear2)
plt.figure(figsize=(5,5))
plt.imshow(data_bilinear3, cmap='gray')
plt.show()