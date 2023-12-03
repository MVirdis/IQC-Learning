# IQC Learning

A non-linear augmentation of linear control policies for uncertain LTI systems, with guarantees of robustness.

Files:
* *control.yml*: the Anaconda python environment with all dependencies used
* *Controller Optimization.ipynb*: a jupyter notebook with some tests of the optimization process
* *models/*: contains the model classes
* *saved_models/*: contains the trained models as torch checkpoint files (.pt)
* *utils/*: contains some utility classes
* *models/*: contains the model classes
* *matlab/*: contains the matlab files for creating the data and the IQC augmentation problem (requires MOSEK and YALMIP)
* *data/*: contains state-space data and initial conditions

References:

	[1] J. Veenman, C. W. Scherer, and H. Köroğlu, “Robust stability and performance analysis based on integral quadratic constraints,” European Journal of Control, vol. 31, pp. 1–32, 2016, doi: 10.1016/j.ejcon.2016.04.004.

    [2] M. Revay, R. Wang, and I. R. Manchester, “Recurrent Equilibrium Networks: Unconstrained Learning of Stable and Robust Dynamical Models,” in 2021 60th IEEE Conference on Decision and Control (CDC), Austin, TX, USA: IEEE, Dec. 2021, pp. 2282–2287. doi: 10.1109/CDC45484.2021.9683054.
