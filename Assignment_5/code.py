import numpy as np
from skimage import io, img_as_float

def k_means(vectors, k, iterations):
    labels = np.full((vectors.shape[0],), -1)
    clusters = np.random.rand(k, 3)
    print("Performing", iterations, "iterations with k as", k)
    for i in range(iterations):
        points = [None for y in range(k)]

        for j, v in enumerate(vectors):
           
            row = np.repeat(v, k).reshape(3, k).T

            labels[j] = np.argmin(np.linalg.norm(row - clusters, axis=1))

            if (points[labels[j]] is None):
                points[labels[j]] = []
                
            points[labels[j]].append(v)

        for y in range(k):
            if (points[y] is not None):
                new_cluster = np.asarray(points[y]).sum(axis=0) / len(points[y])
                clusters[y] = new_cluster
    return (labels, clusters)


img_name = input('File name without ext.: ')
img_path = img_name + '.jpg'
k_val= int(input('What is K (int): '))
iterations = int(input('How many Iterations (int only): '))
image = img_as_float(io.imread(img_path))
dim = image.shape
   
vectors = image.reshape(-1, image.shape[-1])

labels, centroids = k_means(vectors, k_val, iterations)

output_image = np.zeros(vectors.shape)
for i in range(output_image.shape[0]):
    output_image[i] = centroids[labels[i]]

output_image = output_image.reshape(dim)

compressed_name='Compressed_'+str(img_name)+'_'+str(k_val)+'.jpg'
print('Saving as', compressed_name)
io.imsave(compressed_name , output_image)
