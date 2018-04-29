import numpy as np

# predefined dimesnions of input
dim_x = 357;
dim_y = 354;
dim_z = 364;    


def read_image(path):
    """
    Read the 3d image in binary mode from specified path 

    Input:
     - path: Path to input image

    Returns:
     - f: image file object
    """
    input = open(path, mode='rb')
    img_bytes = input.read(size=dim_x * dim_y * dim_z)

    input.close()

    return img_bytes

def find_contact_line_points(img):
    """
    Find only the points that are in the contact line

    Input:
     - img: image as a bytes array

    Returns:
     - contact_points: indices of points in the image that are in the contact line
     - X, Y, Z: indices of contact points as subscrips to 3d array of shape (dim_x, dim_y, dim_z)
    """
    img_array = np.asarray(img)
    contact_points = np.nonzero(img_array == 1)[0]

    # tuple of arrays for contact points
    X, Y, Z = np.nonzero(np.reshape(img_array, (dim_x, dim_y, dim_z)))

    return contact_points, X, Y, Z

def compute_euclidean_distance(contact_points, X, Y, Z):
    """
    Determine the euclidean distance between each point in the contact line
    Input: same as find_contact_line_points() output

    Returns:
     - dist: array of euclidean distances
    """
    dists = np.zeros((1, contact_points.shape[0]))

    for i in range(contact_points.shape[0]):
        dists[i] = 0

        for j in range(contact_points.shape[0]):
            dists[i] += np.sqrt(np.square(X[i] - X[j]) + np.square(Y[i] - Y[j]) + np.square(Z[i] - Z[j]))

    return dists

def add(a, b):
    return a + b

def test():
    assert add(5, 2) == 7

