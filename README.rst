Flow matching for data assimilation
===================================

Installation
============

#. Install ``uv``:

   .. code:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

#. Install Python dependencies using ``uv``:

   .. code:: bash

      uv sync

#. Activate the Python virtual environment:

   .. code:: bash

      source .venv/bin/activate

#. Test your installation:

   .. code:: bash

      pytest tests

#. Edit the ``out_dir`` and ``run_subdir`` fields of the ``Conf`` class in ``src/conf/conf.py`` to the directory where you want the model training output to be saved.

Supplementary Documentation
===========================

* `Hydra <https://hydra.cc/docs/1.3/intro/>`_: Command-line inferface configuration library for configuring the experiments in this project.
* `Hydra ORM <https://github.com/reepoi/hydra-orm>`_: Library for saving experiment configurations to an `SQLite <https://sqlite.org/>`_ database.
* `PyTorch <https://pytorch.org/docs/2.4/index.html>`_: Library for implementing the models.
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/2.5.0/>`_: Library for handling model training.

Running the data assilimation algorithms
========================================

Examples for the running the code are in the ``Examples`` subsection.

Run the command

.. code:: bash

   python src/dafm/main.py dataset=<dataset> model=<model> <other_overrides>...

where:

* ``<dataset>`` is one of:

   * ``DoubleWell``: One-dimensional potential well system from [Bao2024a]_.
   * ``Lorenz63``: Three-dimensional chaotic butterfly attractor system.
   * ``Lorenz96Bao2024ML``: :math:`N`-dimensional chaotic system with parameters from [Bao2024a]_.
   * ``Lorenz96Bao2024EnSF``: :math:`N`-dimensional chaotic system with parameters from [Bao2024b]_.
   * ``Lorenz96H100``: `Lorenz96` with a set of parameters with difficulty rating `H100`.
   * ``Lorenz96H200``: `Lorenz96` with a set of parameters with difficulty rating `H200`.
   * ``Lorenz96H300``: `Lorenz96` with a set of parameters with difficulty rating `H300`.

* ``<model>`` is one of:

   * ``ScoreMatching``: The score matching filter described in [Bao2024a]_ and trains a model every time step.

      * The default parameters are for ``dataset=DoubleWell``.
        Use ``ScoreMatchingLorenz96Bao2024ML`` for ``Lorenz96Bao2024ML``, and ``ScoreMatchingLorenz96`` for ``Lorenz96H***``.

   * ``ScoreMatchingMarginal``: EnSF described in [Bao2024b]_.

      * Variants available: ``ScoreMatchingMarginalBao2024EnSF``

   * ``FlowMatching``: Our flow matching filter that trains a model every time step.

      * The default parameters are for ``dataset=DoubleWell``.
        Use ``FlowMatchingLorenz96Bao2024ML`` for ``Lorenz96Bao2024ML``, and ``FlowMatchingLorenz96`` for ``Lorenz96H***``.

   * ``FlowMatchingMarginal``: Our EnFF methods that approximates the flow matching vector field using a Monte Carlo approximation.

      * Variants available: ``FlowMatchingMarginalConditionalOptimalTransport`` for EnFF-OT and ``FlowMatchingMarginalPreviousPosteriorToPredictive`` for EnFF-F2P.

   * ``FlowMatchingGaussianTarget``: Our flow matching filter that assumes the prediction distribution (Bayesian prior) is Gaussian.

      * Variants available: ``FlowMatchingGaussianTargetConditionalOptimalTransport``

   * ``BootstrapParticleFilter``

      * Variants available: ``BootstrapParticleFilterKuramotoSivashinsky`` for hyperparameters tuned for the Kuramoto-Sivashinsky equation, and ``BootstrapParticleFilterNavierStokes`` for hyperparameters tuned for the Navier-Stokes equation.

   * ``EnsembleKalmanFilterPerturbedObservations``

      * Variants available: ``EnsembleKalmanFilterPerturbedObservationsKuramotoSivashinsky`` for hyperparameters tuned for the Kuramoto-Sivashinsky equation, and ``EnsembleKalmanFilterPerturbedObservationsNavierStokes`` for hyperparameters tuned for the Navier-Stokes equation.

   * ``EnsembleRandomizedSquareRootFilter``: Known as the Ensemble Square Root Filter.

      * Variants available: ``EnsembleRandomizedSquareRootFilterKuramotoSivashinsky`` for hyperparameters tuned for the Kuramoto-Sivashinsky equation, and ``EnsembleRandomizedSquareRootFilterNavierStokes`` for hyperparameters tuned for the Navier-Stokes equation.

   * ``LocalEnsembleTransformKalmanFilter``

      * Variants available: ``LocalEnsembleTransformKalmanFilterKuramotoSivashinsky`` for hyperparameters tuned for the Kuramoto-Sivashinsky equation, and ``LocalEnsembleTransformKalmanFilterNavierStokes`` for hyperparameters tuned for the Navier-Stokes equation.

* ``<other_overrides>...``: Other overrides for the model.
   Add the flag ``-c job`` to the Python command see what can be overridden from the command line.
   Some useful overrides include:

   * Changing the diffusion path:

      .. code:: bash

         model/diffusion_path=VarianceExploding

   * For the flow matching models, changing the guidance vector field approximation:

      .. code:: bash

         model/guidance=<guidance>

      where ``<guidance>`` is one of:

         * ``MonteCarlo*``: the Monte Carlo approximation from section 3.2 of [Feng2025]_.
            * Variants available: ``MonteCarloTargetConditionalOptimalTransport``
         * ``Local*``: the local approximation from section 3.3 of [Feng2025]_.
            * Variants available: ``LocalConstant``

   * Using particle noise perturbation:

      .. code:: bash

         model.use_state_perturbation=true model.state_perturbation_std=0.5

   * Using particle inflation:

      .. code:: bash

         model/inflation_scale=ConstantScale model.inflation_scale.constant=1.01

Examples
--------

The following are example commands to show how to run the code.

.. code:: bash

   # Run EnFF-OT for Kuramoto-Sivashinsky with a grid of size 256
   python src/dafm/main.py dataset=KuramotoSivashinsky dataset.state_dimension=256 model=FlowMatchingMarginalConditionalOptimalTransport model/guidance=LocalConstant model.guidance.schedule.constant=1 model.sampling_time_step_count=20
   # Run EnFF-F2P for Navier-Stokes with a grid of size 64x64
   python src/dafm/main.py rng_seed={rng_seed} dataset=NavierStokesDim64 model=FlowMatchingMarginalPreviousPosteriorToPredictive model/guidance=LocalConstant model.guidance.schedule.constant=.2 model.sampling_time_step_count=50

Running data assimilation experiments with high-dimensional systems
===================================================================

By default, the true system state and observation trajectories are computed to the terminal time and stored on the specified device (see ``Conf.device`` in ``src/conf/conf.py``);
however, for high-dimensional systems, this can require more RAM than is available when the ``Conf.device == 'cuda'``.
To work around this, we introduce the configuration setting in ``Dataset.trajectory_stored_on_gpu_max_state_dimension`` in ``src/conf/datsets.py``.
When ``Dataset.state_dimension > Dataset.trajectory_stored_on_gpu_max_state_dimension``, we store these trajectories as if ``Conf.device == 'cpu'``, then move the PyTorch tensors to ``Conf.device == 'cuda'`` when necessary (e.g., when sampling from a generative model).

Also, by default, all the particle states are saved to a `.parquet` file once the experiment is complete.
For high-dimensional systems, this can be many gigabytes of data.
Use ``Dataset.save_only_mean_std`` to save the mean and standard deviation of the particles for each time step and dimension.

.. warning::

   Due to a bug in ``hydra-orm``, the configuration settings mentioned here must be edited in their respective Python files.
   Command line overrides for these settings will be ignored.

Running data assimilation experiments in parallel
=================================================

.. warning::

   On network file systems (NFS), starting multiple processes running this code can corrupt the SQLite database storing the experiment configurations.
   See question (5) of the `SQLite FAQs <https://sqlite.org/faq.html>`_.
   See the Preflight section below to see how to ensure the experiment configurations are written to the database serially.

Using `GNU parallel <https://www.gnu.org/software/parallel/>`_, multiple experiments can be run in parallel.

.. code:: bash

   parallel --eta --header : python src/dafm/main.py <override_1>={<param_1>} <override_2>={<param_2>} ... ::: <param_1> <p1value_1> <p1value_2> ... ::: <param_2> <p2value_1> <p2value_2> ...


Preflight
---------

To ensure that experiment configurations are saved to the database serially, run GNU parallel command with ``-j 1`` and the Python command with ``-c job``.

.. code:: bash

   parallel -j 1 --eta --header : python src/dafm/main.py -c job <override_1>={<param_1>} <override_2>={<param_2>} ... ::: <param_1> <p1value_1> <p1value_2> ... ::: <param_2> <p2value_1> <p2value_2> ...

Once this command has finished, all the experiment configurations have been saved.
Next, run the first GNU parallel command to begin running the experiments in parallel.

References
==========

.. [Bao2024a] `F. Bao, Z. Zhang, and G. Zhang, "A score-based filter for nonlinear data assimilation," Journal of Computational Physics, vol. 514, p. 113207, Oct. 2024, doi: 10.1016/j.jcp.2024.113207. <https://www.sciencedirect.com/science/article/pii/S002199912400456X>`_
.. [Bao2024b] `F. Bao, Z. Zhang, and G. Zhang, "An ensemble score filter for tracking high-dimensional nonlinear dynamical systems," Computer Methods in Applied Mechanics and Engineering, vol. 432, p. 117447, Dec. 2024, doi: 10.1016/j.cma.2024.117447. <https://www.sciencedirect.com/science/article/pii/S0045782524007023>`_
.. [Feng2025] `R. Feng, T. Wu, C. Yu, W. Deng, and P. Hu, "On the Guidance of Flow Matching," Feb. 04, 2025, arXiv: arXiv:2502.02150. doi: 10.48550/arXiv.2502.02150. <http://arxiv.org/abs/2502.02150>`_
