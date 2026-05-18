Theory
======

The package evaluates perturbations acting on matrix population models of the
form

.. math::

   \mathbf{n}_{t+1} = \mathbf{A}_t \mathbf{n}_t,

where :math:`\mathbf{n}_t` is the population vector at time :math:`t` and
:math:`\mathbf{A}_t` is the projection matrix.

Perturbation regimes
--------------------

Each perturbation regime is defined by three quantities:

- ``magnitude``: proportional reduction applied to the selected matrix entries;
- ``duration``: number of consecutive time steps during which a perturbation is active;
- ``period``: number of time steps between the onset of consecutive perturbation events.

A duration of one time step corresponds to a pulse perturbation, whereas longer
durations represent sustained perturbations over multiple projection intervals.

Population reduction
--------------------

For a given perturbation regime, population reduction is defined as

.. math::

   \rho = 100 \left(1 - \frac{N_{\mathrm{pert}}(T)}
   {N_{\mathrm{base}}(T)}\right),

where :math:`N_{\mathrm{pert}}(T)` is the final population size under the
perturbed dynamics and :math:`N_{\mathrm{base}}(T)` is the final population size
under the unperturbed baseline dynamics.

Integrated vulnerability
------------------------

Integrated vulnerability is computed as the mean population reduction across the
simulated perturbation space:

.. math::

   \Phi = \langle \rho(m,d,p) \rangle_{\Omega},

where :math:`m` is perturbation magnitude, :math:`d` is perturbation duration,
:math:`p` is perturbation period, and :math:`\Omega` is the set of simulated
perturbation regimes.
