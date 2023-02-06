from registration import registration
from get_pointcloud import point_cloud_data

import numpy as np
from typing import Tuple
from math import sqrt
from numpy import linalg as LA

class registration_iasd(registration):

    def __init__(self, scan_1: np.array((..., 3)), scan_2: np.array((..., 3))) -> None:

        # inherit all the methods and properties from registration
        super().__init__(scan_1, scan_2)

        return

    def compute_pose(self,correspondences: dict) -> Tuple[np.array, np.array]:
        """compute the transformation that aligns two
        scans given a set of correspondences

        :param correspondences: set of correspondences
        :type correspondences: dict
        :return: rotation and translation that align the correspondences
        :rtype: Tuple[np.array, np.array]
        """
        ## Get the point clouds of the 1st and 2nd scan from correspondences dict
        p = []
        q = []
        
        for pc_info in correspondences.values():
            p.append(pc_info['point_in_pc_1'])
            q.append(pc_info['point_in_pc_2'])  
        
        ## Get the center points
        center_pc1=np.sum(p,axis=0)/(len(p))
        center_pc2=np.sum(q,axis=0)/(len(q))

        ## Frame alignement 
        P=p-center_pc1
        Q=q-center_pc2
        
        ## Get matrix A
        A = np.dot(Q.T,P)

        ## Get U, s and Vt using SVD
        U, s, Vt = np.linalg.svd(A, full_matrices=True)
        B=np.eye(3)
        B[2,2]=np.linalg.det(np.dot(U,Vt))

        ## Get output rotation and translation
        R_out=np.linalg.multi_dot([U,B,Vt])
        t_out=center_pc2-np.dot(R_out,center_pc1)

        return (R_out,t_out)



    def find_closest_points(self) -> dict:
        """Computes the closest points in the two scans.
        There are many strategies. We are taking all the points in the first scan
        and search for the closes in the second. This means that we can have > than 1 points in scan
        1 corresponding to the same point in scan 2. All points in scan 1 will have correspondence.
        Points in scan 2 do not have necessarily a correspondence.

        :param search_alg: choose the searching option
        :type search_alg: str, optional
        :return: a dictionary with the correspondences. Keys are numbers identifying the id of the correspondence.
                Values are a dictionaries with 'point_in_pc_1', 'point_in_pc_2' identifying the pair of points in the correspondence.
        :rtype: dict
        """

        dict = {}
        dict2 = {}
        count = 0

        for i in self.scan_1[:,]:
            count += 1
            aux = np.linalg.norm(i-self.scan_2[:,],axis=1) ## For each point of scan1, computes the norm between that point and each point of scan2 
            ind = np.argmin(aux) ## Gets the minimum norm value in order to find the closest point in scan2 to every point in scan1
            key = {
                'point_in_pc_1': i, ## Point in scan1
                'point_in_pc_2': self.scan_2[ind,], ## Closest point in scan2 to point in scan1
                'dist2': float(aux[ind]) ## Distance between those two points
            }
            dict2 = {'key'+str(count):key} ## Creates the dictionary of correspondences
            dict.update(dict2)

        return dict


class point_cloud_data_iasd(point_cloud_data):

    def __init__(
            self,
            fileName: str,
            ) -> None:
            
        super().__init__(fileName)

        return


    def load_point_cloud(self, file: str) -> bool:
        """Loads a point cloud from a ply file

        :param file: source file
        :type file: str
        :return: returns true or false, depending on whether the method
        the ply was OK or not
        :rtype: bool
        """
        
        line_count = 0
        find_string = 'end_header'
        find_vertex = 'element vertex'
        find_x = 'property float x'
        find_y = 'property float y'
        find_z = 'property float z'
        valid = 0
        order = [0,0,0]
        pc = {}
        pc_aux = {}
        with open(file, "r") as filename:
            for line in filename:

                ## Starts to read the file from the end of the header
                if find_string in line:
                    values_file = filename.readlines()  

                ## Saves the order in which coordinates x, y and z are read from the file 
                ## and validate data dimension
                if find_x in line:
                    order[0] = valid
                    valid += 1

                if find_y in line:
                    order[1] = valid
                    valid += 1

                if find_z in line:
                    order[2] = valid
                    valid += 1
                
                ## Checks the number of points in the pointcloud indicated in the file 
                if find_vertex in line:
                    n_vertex=[int(word) for word in line.split() if word.isdigit()]

            for line in values_file:

                ## Finish pointcloud read
                if line_count==n_vertex[0]:
                    break

                line_count += 1
                vector = np.array(line.rstrip().split()).astype(np.float) ## Creates an array with the coordinates of the point
                vector = vector[:3].T

                
                if len(vector) < 3: ## Data size validation
                    valid = len(vector)
                else:
                    vector = vector[order[:]] ## Vector with the correct coordinate order of the point: [x y z]
                pc_aux = {'key'+str(line_count): vector} ## Filling up the dictionary with the keys and their correspondent values 
                pc.update(pc_aux)

        filename.close()
        self.data = pc
        
        ## Checks if the file has 3 coordinates for each point
        if valid < 3:
             return False
                    
        return True