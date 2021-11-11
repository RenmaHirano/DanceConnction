from PIL import Image
import numpy as np
from numpy.lib.type_check import imag
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

images = np.array(Image.open("OutputImgs/img (1).jpg").convert("P"))
images = images / 255
images = images.flatten()

for i in range(1821): #1821
    addImage = np.array(Image.open("OutputImgs/img (" + str(i + 2) + ").jpg").convert("L"))
    addImage = addImage / 255
    imageVector = addImage.flatten()
    images = np.vstack([images, imageVector])

pca = PCA(n_components=9)
pca.fit(images)
value = pca.transform(images)

for i in range(9):
    out = pca.components_[i].reshape(254,254)
    outputImage = Image.fromarray(np.uint8(out*255))
    outputImage = outputImage.convert("P")
    outputImage.save("outputImage_" + str(i) + ".png")

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Helix", size = 20)
ax.plot(value[:,6],value[:,7],value[:,8], color = "green")
plt.show()

f = open('sample.txt', 'w')
f.writelines(np.uint8(out*255))
f.writelines("\n")
f.close()
