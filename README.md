# EAPPnP (Efficient Anisotropic Procrutes PnP)
Perspective n point method for 3D structure with unknown stretching 

# Method and Purpose
This algorithm is proposed to solve a generalized version of the well-known Perspective-n-Point(PnP) problem that tries to recover not only the pose but also the scaling vector assuming the known 3D structure has undergone some uneven stretching along the 3 axis in space. The method is based on a state-of-the-art PnP solution called Efficient Procrutes PnP (EPPnP) (You see where the name comes from, see http://digital.csic.es/bitstream/10261/127304/6/Outlier-Rejection.pdf for details). To solve the generalized PnP problem we modified the Efficient Procrutes PnP to solve the anisotropic Procrutes problem instead of the original isotropic Procrutes problem. The additional 2 degree of freedom (scaling along 3 axis, but up to a scale) can be resolve using the same procedure as the original paper.

# Benchmark
We first want to make sure our implementation of EPPnP is correct and comparable with the original matlab implementation, then we can check the performance of EAPPnP on the ordinary PnP problem. We replicate the experiments in the original paper and find the result a little bit worse than the number reported in the orginal paper (but similar to numbers reported by others who also replicated the experiments, see https://arxiv.org/pdf/1607.08112.pdf).

## Ordinary PnP
### Execution time
<img src="https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_time.png" width="400" height="300">

### Error with varying point set size
<img src="https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_mix_gaussian_rot.png" width="400" height="300"><img src="https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_mix_gaussian_trans.png" width="400" height="300">

### Error with varying noise level
<img src="https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_max_gaussian_rot.png" width="400" height="300"><img src="https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_max_gaussian_trans.png" width="400" height="300">

## Generalized PnP
In this setting, the 3D structure is scaled along y and z axis randomly within the range from 0.5 to 2. Notice that not all axis has been randomly scaled due to the property of monocular camera that everything is up to a scale factor.
### Error with varying point set size
<img src="https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_mix_gaussian_rot_aniso.png" width="400" height="300"><img src="https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_mix_gaussian_trans_aniso.png" width="400" height="300">

### Error with varying noise level
<img src="https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_max_gaussian_rot_aniso.png" width="400" height="300"><img src="https://github.com/derleeG/EAPPnP/blob/master/fig/Figure_max_gaussian_trans_aniso.png" width="400" height="300">

(The error of EPPnP method under this setting is too big to show on the same plot)
