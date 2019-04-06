 # Preconditioning Kaczmarz method by sketching
 
 This repository presents the source code and examples of using *Preconditioned Kaczmarz method*, where preconditioner is constructed from QR decomposition of the sketched system matrix. We demonstrate numerically that even a small subset of rows of the system matrix can give sufficiently good preconditioner. The examples of using the presented method are also available here, in the folder ```examples```.  
 
 ## Examples
 
 We provide numerical experiments to demonstrate the performance of the proposed approach:
 
 - Random overdetermined ill-conditioned and consistent linear systems ([GitHub](./examples/random_data.ipynb), [Nbviewer](https://nbviewer.jupyter.org/github/amkatrutsa/preckacz/blob/master/examples/random_data.ipynb))
 - Kernel regression with explicit feature map via Random Fourier Features ([GitHub](./examples/kernel_regression_via_rff.ipynb), [Nbviewer](https://nbviewer.jupyter.org/github/amkatrutsa/preckacz/blob/master/examples/kernel_regression_via_rff.ipynb))
 - Image reconstruction from the tomography data ([GitHub](./examples/tomography.ipynb), [Nbviewer](https://nbviewer.jupyter.org/github/amkatrutsa/preckacz/blob/master/examples/tomography.ipynb)) The used images and matrices are stored in ```data``` folder.
 
 ## Citing
 
If you use this research in your work, we kindly ask you to cite [the paper](https://arxiv.org/pdf/1903.01806.pdf)

```
@article{katrutsa2019preconditioning,
  title={Preconditioning Kaczmarz method by sketching},
  author={Katrutsa, Alexandr and Oseledets, Ivan},
  journal={arXiv preprint arXiv:1903.01806},
  year={2019}
}
```
