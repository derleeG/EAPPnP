# EAPPnP (Efficient Anisotropic Procrutes PnP)
Perspective n point method for 3D structure with unknown stretching 

# Method and Purpose
This algorithm is proposed to solve a generalized version of the well-known Perspective-n-Point(PnP) problem that tries to recover not only the pose but also the scaling vector assuming the known 3D structure has undergone some uneven stretching along the 3 axis in space. The method is based on a state-of-the-art PnP solution called Efficient Procrutes PnP (EPPnP) (You see where the name comes from, see http://digital.csic.es/bitstream/10261/127304/6/Outlier-Rejection.pdf for details). To solve the generalized PnP problem we modified the Efficient Procrutes PnP to solve the anisotropic Procrutes problem instead of the original isotropic Procrutes problem. The additional 2 degree of freedom (scaling along 3 axis, but up to a scale) can be resolve using the same procedure as the original paper.

# Benchmark
We first want to make sure our implementation of EPPnP is correct and comparable with the original matlab implementation, then we can check the performance of EAPPnP on the ordinary PnP problem. We replicate the experiments in the original paper and find the result a little bit worse than the number reported in the orginal paper (but similar to numbers reported by others who also replicated the experiments, see https://arxiv.org/pdf/1607.08112.pdf).

## Ordinary PnP
### execution time
![alt text](https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_time.png "Execution time plot")

### Error with varying point set size
![alt text](https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_mix_gaussian_rot.png "Fix noise rotation error plot") 
![alt text](https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_mix_gaussian_trans.png "Fix noise translation error plot")

### Error with varying noise level
![alt text](https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_max_gaussian_rot.png "Fix noise rotation error plot") 
![alt text](https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_max_gaussian_trans.png "Fix noise translation error plot")
