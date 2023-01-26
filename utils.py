import numpy as np
from numba import jit



def eigen_cleaner(eigenvalues,eigenvectors,lambda_plus):

    N=len(eigenvalues)
    cleaned_correlation_matrix = np.zeros((N, N))

    noisy_eigen = eigenvalues<=lambda_plus                     # these eigenvalues come from the seemingly random bulk
    delta=np.mean(eigenvalues[noisy_eigen])                    # delta is their average, so as to conserver the trace of C

    lambdas_clean= np.copy(eigenvalues)
    lambdas_clean[noisy_eigen]=delta
    
      
    
    v_m=np.matrix(eigenvectors)

    for i in range(N-1):
        a = lambdas_clean[i] * np.dot(v_m[i,].T,v_m[i,])
        cleaned_correlation_matrix=cleaned_correlation_matrix+ a
        
    cleaned_correlation_matrix = np.real(cleaned_correlation_matrix)
    #np.fill_diagonal(cleaned_correlation_matrix,1)

    
    return np.clip(cleaned_correlation_matrix, -1, 1)



def compute_clean_correlation_matrix(raw_correlations, n, T):
    spectral_vals, spectral_vectors = np.linalg.eig(raw_correlations)
    lambda_plus = (1. + np.sqrt(n * 1. /T))**2
    cleaned_correlations = eigen_cleaner(spectral_vals, spectral_vectors, lambda_plus=lambda_plus)
    return cleaned_correlations




    
"""
 data =  ((data.diff()/data)).fillna(0)
    raw_correlations = data.corr().fillna(0)
    spectral_vals = np.linalg.eig(raw_correlations)
    lambda_plus = (1. + np.sqrt(data.shape[1] * 1. / data.shape[0]))**2
    cleaned_correlations = eigen_cleaner(*spectral_vals, lambda_plus=lambda_plus)
    return raw_correlations, cleaned_correlations

"""
    