Data Assimilation with Flow Matching: The Ensemble Flow Filter
==============================================================

Code for replicating the results in `Flow Matching for Efficient and Scalable Data Assimilation <https://arxiv.org/abs/2508.13313>`_.

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

#. Edit the ``out_dir`` and ``run_subdir`` fields of the ``Conf`` class in ``src/conf/conf.py`` to the directory where you want the output of every experiment to be saved.

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

   * ``Lorenz96Bao2024EnSF``: :math:`N`-dimensional chaotic system with parameters from [Bao2024b]_.
   * ``KuramotoSivashinsky``: 1-dimensional chaotic Kuramoto-Sivashinsky PDE.
   * ``NavierStokesDim256``: Navier-Stokes PDE with periodic boundary conditions discretized on a :math:`256x256` grid.

* ``<model>`` is one of:

   * ``ScoreMatchingMarginal``: EnSF described in [Bao2024b]_.

      * Variants available: ``ScoreMatchingMarginalBao2024EnSF``

   * ``FlowMatchingMarginal``: Our EnFF methods that approximates the flow matching vector field using a Monte Carlo approximation.

      * Variants available: ``FlowMatchingMarginalConditionalOptimalTransport`` for EnFF-OT and ``FlowMatchingMarginalPreviousPosteriorToPredictive`` for EnFF-F2P.

   * ``BootstrapParticleFilter``

   * ``EnsembleKalmanFilterPerturbedObservations``

   * ``EnsembleKalmanFilterPerturbedObservationsIterative``

   * ``EnsembleRandomizedSquareRootFilter``: Known as the Ensemble Square Root Filter.

   * ``LocalEnsembleTransformKalmanFilter``

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

   # Run EnFF-F2P for Kuramoto-Sivashinsky with a grid of size 1024
   python src/dafm/main.py dataset=KuramotoSivashinsky model=FlowMatchingMarginalPreviousPosteriorToPredictive model/guidance=LocalConstant model.guidance.schedule.constant=0.005 model.diffusion_path.sigma_min=1e-3 model.sampling_time_step_count=5
   # Run EnFF-F2P for Navier-Stokes with a grid of size 256x256
   python src/dafm/main.py dataset=NavierStokesDim256 model=FlowMatchingMarginalPreviousPosteriorToPredictive model/guidance=LocalConstant model.guidance.schedule.constant=0.001 model.diffusion_path.sigma_min=1e-3 model.sampling_time_step_count=10

For more examples, see the following bash scripts:

   * ``tune.sh``: Runs a hyperparameter sweep for EnFF-OT, EnFF-F2P, and EnSF.

   * ``tune_classical.sh``: Runs a hyperparameter sweep for the classical methods (e.g., EnKF).

   * ``test_best.sh``: Evaluates EnFF-OT, EnFF-F2P, and EnSF using the best hyperparameters.

   * ``test_best_classical_comparison.sh``: Evaluates all methods using the best hyperparameters for the datasets used in the comparison with classical methods.

To run these scripts, install `GNU parallel <https://www.gnu.org/software/parallel/>`_.
Once installed, replace ``--eta -j 1`` with ``--dry-run`` in the bash scripts to generate many example commands.

Processing experiment output
============================

We provide Jupyter notebooks in the notebooks directory to process the experiment output:

   * ``tune.ipynb``: Notebook for compiling the results of a hyperparameter sweep for EnFF-OT, EnFF-F2P, and EnSF, or a hyperparameter sweep for the comparison with classical methods.
     See ``tune.sh`` and ``tune_classical.sh`` to run these hyperparameter sweeps.
     It saves a CSV file containing the best hyperparameters in the ``sweeps`` directory.

   * ``logged_metrics.ipynb``: Notebook for producing Figure 2.
     See ``test_best.sh`` to produce the data for this figure.

   * ``sensitivity.ipynb``: Notebook for producing the ablation study figure for EnFF-OT and EnFF-F2P (Figure 6).
     See ``tune.sh`` to produce the data for this figure.

   * ``classical_comparison.ipynb``: Notebook for producing Figure 5.
     See ``test_best_classical_comparison.sh`` to produce the data for this figure.

   * ``datasets_*.ipynb``: Notebooks for visualizing the dynamical systems used in the paper.

   * ``trajectories_*.ipynb``: Notebooks for visualizing the estimated dynamical system states produced by each model.
     Set ``save_data=True`` in ``src/dafm/main.py`` on line 86 to save the estimated states before running the model to save the estimated states.

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

.. [Bao2024b] `F. Bao, Z. Zhang, and G. Zhang, "An ensemble score filter for tracking high-dimensional nonlinear dynamical systems," Computer Methods in Applied Mechanics and Engineering, vol. 432, p. 117447, Dec. 2024, doi: 10.1016/j.cma.2024.117447. <https://www.sciencedirect.com/science/article/pii/S0045782524007023>`_
.. [Feng2025] `R. Feng, T. Wu, C. Yu, W. Deng, and P. Hu, "On the Guidance of Flow Matching," Feb. 04, 2025, arXiv: arXiv:2502.02150. doi: 10.48550/arXiv.2502.02150. <http://arxiv.org/abs/2502.02150>`_
