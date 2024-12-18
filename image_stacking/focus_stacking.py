# %% [markdown]
### Focus Stacking
# This project will explore focus stacking across multiple images to produce an increase Depth of Field (DOF)
# in a single output image.

# %% [markdown]
## Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob

# %% [markdown]
## Sandbox
# - **Remember!** OpenCV imports images in BGR (blue, green, red)

# %%
# pic01 = cv2.imread('/home/unitx/wabbit_playground/image_stacking/autofocus/Image__2024-12-17__15-20-46.png')
# pic02 = cv2.imread('/home/unitx/wabbit_playground/image_stacking/autofocus/Image__2024-12-17__15-21-12.png')
# pic03 = cv2.imread('/home/unitx/wabbit_playground/image_stacking/autofocus/Image__2024-12-17__15-22-02.png')

# pic01_cvt = cv2.cvtColor(pic01, cv2.COLOR_BGR2RGB)
# pic02_cvt = cv2.cvtColor(pic02, cv2.COLOR_BGR2RGB)
# pic03_cvt = cv2.cvtColor(pic03, cv2.COLOR_BGR2RGB)

# imS = cv2.resize(pic01, (960, 540))   

# cv2.resizeWindow('piccc', 600, 400)
# cv2.imshow('hi', imS)
# cv2.waitKey()
# cv2.destroyAllWindows()
# # %%
# images = [cv2.imread('/home/unitx/wabbit_playground/image_stacking/autofocus/Image__2024-12-17__15-20-46.png'),
#         cv2.imread('/home/unitx/wabbit_playground/image_stacking/autofocus/Image__2024-12-17__15-21-12.png'),
#         cv2.imread('/home/unitx/wabbit_playground/image_stacking/autofocus/Image__2024-12-17__15-22-02.png')]

# images_cvt = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

# laplacians = [cv2.Laplacian(img, cv2.CV_64F) for img in images_cvt]
# # print(np.max(laplacians))
# variance = [np.var(lap) for lap in laplacians]

# # %%
# laplacians = (laplacians - np.min(laplacians)) / (np.max(laplacians) - np.min(laplacians))
# threshold = 0
# masks = [lap > threshold for lap in laplacians]

# result = np.zeros_like(images_cvt[0])
# for img, mask in zip(images_cvt, masks):
#     result = np.where(mask, img, result)

# print(result.shape)
    
# plt.imshow(result)
# plt.show()

# %%
image_names = glob('/home/unitx/wabbit_playground/image_stacking/autofocus/*.png')
images = [plt.imread(img) for img in image_names]
images = np.array(images, np.uint8)
# %%
laplacians = [cv2.Laplacian(img, cv2.CV_64F) for img in images]
# print(np.max(laplacians))
variance = [np.var(lap) for lap in laplacians]

# %%
laplacians = (laplacians - np.min(laplacians)) / (np.max(laplacians) - np.min(laplacians))
threshold = 0
masks = [lap > threshold for lap in laplacians]
result = np.zeros_like(images[0])
for img, mask in zip(images, masks):
    result = np.where(mask, img, result)


# %%
plt.imshow(result)
plt.show()