import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    eigenvalue = 0
    eigenvector = np.ones(data.shape[1])
    
    for _ in range(num_steps):
        eigenvector = data.dot(eigenvector) / np.linalg.norm(data.dot(eigenvector))
        eigenvalue = float((eigenvector.T.dot(data).dot(eigenvector)) / eigenvector.T.dot(eigenvector))

    return eigenvalue, eigenvector