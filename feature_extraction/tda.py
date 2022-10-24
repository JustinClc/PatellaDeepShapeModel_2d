import gudhi as gd
import gudhi.representations
from tqdm import tqdm
import numpy as np

def tda_feat_extract(shape_data, repre_model_params:dict):
    X_tda = []
    for ID in tqdm(list(shape_data)):
        
        # Calculating single tda representation vector
        def tda_single_feat(repre, multi_pers, kth_pers):
            acX = gd.AlphaComplex(points=shape_data[ID]).create_simplex_tree()
            dgmX = acX.persistence()
            #gd.plot_persistence_diagram(dgmX)
            
            if repre == 'poly':
                CP = gd.representations.vector_methods.ComplexPolynomial()
            elif repre == 'sil':
                CP = gd.representations.vector_methods.Silhouette(resolution=100)
            elif repre == 'entropy':
                CP = gd.representations.vector_methods.Entropy(mode='scalar')
            elif repre == 'landscape':
                CP = gd.representations.vector_methods.Landscape() 
            elif repre == 'pi':
                CP = gd.representations.vector_methods.PersistenceImage(bandwidth=1.0)

            if multi_pers == True:
                persistence_0th = acX.persistence_intervals_in_dimension(0)
                persistence_1st = acX.persistence_intervals_in_dimension(1)
                persistence_0th[persistence_0th == np.inf] = 0
                persistence_1st[persistence_1st == np.inf] = 0

                # Representation of 0th persistence
                CP.fit([persistence_0th])
                cp_0 = CP.transform([persistence_0th])
                cp_0 = cp_0.real.flatten()

                # Representation of 1st persistence
                CP.fit([persistence_1st])
                cp_1 = CP.transform([persistence_1st])
                cp_1 = cp_1.real.flatten()

                single_tda = np.hstack((cp_0, cp_1)).flatten()

            else:
                persistence = acX.persistence_intervals_in_dimension(kth_pers)
                persistence[persistence == np.inf] = 0
                CP.fit([persistence])
                cp = CP.transform([persistence])
                single_tda = cp.real.flatten()
                
            return single_tda
        
        # Stack all tda representations horizontally to form a multi-tda vector
        if len(repre_model_params['representation']) > 1:
            b = []
            for repre in list(zip(*repre_model_params.values())):
                a = tda_single_feat(*repre)
                b.append(a)
            multi_tda = np.hstack(b)
            X_tda.append(multi_tda)
            
        else:
            single_tda = tda_single_feat(*list(zip(*repre_model_params.values()))[0])
            X_tda.append(single_tda)
            
    return np.array(X_tda)
