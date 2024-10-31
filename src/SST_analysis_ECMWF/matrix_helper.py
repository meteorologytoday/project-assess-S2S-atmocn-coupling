import scipy.sparse as sparse
import numpy as np



def constructSubspaceWith(mask):


    N = len(mask)
    N_keep = np.sum(mask)

    row_ind = np.zeros((N_keep,), dtype=int)
    col_ind = np.zeros((N_keep,), dtype=int)
    data = np.ones((N_keep,))

    ind_cnt = 0
    for i in range(N):
        
            
        if not mask[i]:
            continue
       
        row_ind[ind_cnt] = ind_cnt
        col_ind[ind_cnt] = i

        ind_cnt +=1


    return sparse.csc_matrix((data, (row_ind, col_ind)), shape=(N_keep, N)) 
        
            
            
            
            
if __name__ == "__main__":
    
    x = np.arange(10, dtype=float)
    
    x[[2, 5, 7]] = np.nan

    M = constructSubspaceWith(np.isfinite(x))
   
    print("x = ", x)
    print("M = ", M)
    print("M @ x = ", M @ x)
    print("M^T @ M @ x = ", M.T @ M @ x)
