Usage
=====

Single perturbation trajectory
------------------------------

.. code-block:: python

   import numpy as np
   from demovuln import MatrixPopulationModel, simulate_dynamics

   A = np.array([
       [0.0, 2.0],
       [0.4, 0.7],
   ])

   model = MatrixPopulationModel(A)

   result = simulate_dynamics(
       model,
       target="adult_survival",
       magnitude=0.25,
       duration=1,
       period=3,
       t_max=50,
       recovery_steps=10,
   )

   print(result.reduction)
   print(result.abundance)

Full perturbation grid
----------------------

.. code-block:: python

   import numpy as np
   from demovuln import MatrixPopulationModel, PerturbationGrid, run_grid

   A = np.array([
       [0.0, 2.0],
       [0.4, 0.7],
   ])

   model = MatrixPopulationModel(A)

   grid = PerturbationGrid(
       magnitudes=np.linspace(0, 1, 11),
       durations=[0, 1, 2, 3],
       periods=[1, 2, 3, 5, 10],
   )

   out = run_grid(
       model,
       target="adult_survival",
       grid=grid,
       t_max=50,
       recovery_steps=10,
   )

   print(out.vulnerability)
   print(out.table.head())

Explicit demographic targets
----------------------------

By default, adult stages are inferred as source-stage columns with at least one
fecundity entry, and juvenile stages are inferred as the remaining source-stage
columns. These definitions can be specified explicitly:

.. code-block:: python

   model = MatrixPopulationModel(
       A,
       adult_stages=[1],
       juvenile_stages=[0],
   )

Custom perturbation masks
-------------------------

Users can also define custom perturbation targets by passing a Boolean matrix
with the same shape as the projection matrix.

.. code-block:: python

   custom_mask = np.array([
       [False, False],
       [True, False],
   ])

   result = simulate_dynamics(
       model,
       target="custom",
       custom_mask=custom_mask,
       magnitude=0.5,
       duration=1,
       period=3,
       t_max=50,
   )
