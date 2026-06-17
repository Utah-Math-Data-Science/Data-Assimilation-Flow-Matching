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

      uv run pytest tests

#. Edit the ``out_dir`` and ``run_subdir`` fields of the ``Conf`` class in ``src/conf/conf.py`` to the directory where you want the output of every experiment to be saved.


Supplementary Documentation
===========================

* `Hydra <https://hydra.cc/docs/1.3/intro/>`_: Command-line interface configuration library for configuring the experiments in this project.
* `Hydra ORM <https://github.com/reepoi/hydra-orm>`_: Library for saving experiment configurations to an `SQLite <https://sqlite.org/>`_ database.
* `PyTorch <https://pytorch.org/docs/2.4/index.html>`_: Library for implementing the models.
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/2.5.0/>`_: Library for handling model training.


Running the data assimilation algorithms
========================================

Examples for running the code are in the ``Examples`` subsection.

Run the command

.. code:: bash

   python src/dafm/main.py +experiment=<experiment> filter=<filter> <other_overrides>...

where:

* ``<experiment>`` is one of:

   * ``Lorenz63Spantini2022``
   * ``Lorenz96Bao2024``
   * ``KuramotoSivashinsky``
   * ``NavierStokesDim16Slow`` / ``NavierStokesDim64Slow`` / ``NavierStokesDim256``

* ``<filter>`` is one of:

   * ``EnsembleFlowFilter``: EnFF methods.

      * Typical overrides: ``filter/prob_path=<prob_path>`` and ``filter/guidance=LocalConstant``.

      * ``<prob_path>`` is usually one of ``ConditionalOptimalTransport`` or ``FilteringToPredictive``.

   * ``EnsembleScoreFilter``: EnSF described in [Bao2024]_.

      * Typical override: ``filter/prob_path=Bao2024EnsembleScoreMatching``.

   * ``BootstrapParticleFilter``

   * ``EnsembleKalmanFilterPerturbedObservations``

   * ``EnsembleKalmanFilterPerturbedObservationsIterative``

   * ``EnsembleRandomizedSquareRootFilter``: Known as the Ensemble Square Root Filter.

   * ``LocalEnsembleTransformKalmanFilter``

* ``<other_overrides>...``: Other overrides for the experiment.
   Add the flag ``-c job`` to the Python command to see what can be overridden from the command line.
   Some useful overrides include:

   * For EnFF and EnSF, selecting a probability path:

      .. code:: bash

         filter/prob_path=<prob_path>

   * For EnFF, changing the guidance vector field approximation:

      .. code:: bash

         filter/guidance=<guidance>

      where ``<guidance>`` is one of:

         * ``Local*``: the local approximation from section 3.3 of [Feng2025]_.
            * Variants available: ``LocalConstant``

   * Changing the number of sampling steps used by EnFF/EnSF:

      .. code:: bash

         filter.sampling_time_step_count=10

   * Reusing a precomputed reference filter:

      .. code:: bash

         setting.reference_filter=<reference_filter_alt_id>

Examples
--------

The following are example commands to show how to run the code.
The hyperparameter override values in these examples were found using Optuna sweeps for the corresponding settings.

.. code:: bash

   # Run EnFF-F2P for Kuramoto-Sivashinsky with ATan observations
   python src/dafm/main.py +experiment=KuramotoSivashinsky filter=EnsembleFlowFilter filter/prob_path=FilteringToPredictive filter/guidance=LocalConstant filter.sampling_time_step_count=5 filter.prob_path.sigma_min=0.005263515825015734 filter.guidance.schedule.constant=0.006935310286542734

   # Run EnFF-F2P for Navier-Stokes with ATan observations
   python src/dafm/main.py +experiment=NavierStokesDim256 filter=EnsembleFlowFilter filter/prob_path=FilteringToPredictive filter/guidance=LocalConstant filter.sampling_time_step_count=10 filter.prob_path.sigma_min=0.00027422730673253964 filter.guidance.schedule.constant=0.0010076278359930006

   # Run EnSF for Kuramoto-Sivashinsky with identity observations
   python src/dafm/main.py +experiment=KuramotoSivashinsky filter=EnsembleScoreFilter filter/prob_path=Bao2024EnsembleScoreMatching filter.sampling_time_step_count=5 filter.prob_path.epsilon_alpha=0.8889480892914582 filter.prob_path.epsilon_beta=0.07766590666645484 "setting.observes=[{_target_:conf.observe.ObserveIdentity,order:0}]" setting.obs_noise_std=.5

   # Run Optuna sweep for EnFF on Kuramoto-Sivashinsky
   python src/dafm/main_optuna.py +experiment=KuramotoSivashinsky filter=EnsembleFlowFilter filter/prob_path=FilteringToPredictive filter/guidance=LocalConstant filter.sampling_time_step_count=5 "optuna_trial_params=[{override:filter.prob_path.sigma_min,type:float,min:1e-5,max:1,log:true},{override:filter.guidance.schedule.constant,type:float,min:1e-3,max:10,log:true}]"

   # Re-run an existing config with a new rng seed
   python src/dafm/rerun_from_alt_id.py --base-alt-id <alt_id> --rng-seed <rng_seed> --save-ensemble-stats true

For more examples, see the following bash scripts:

   * ``sweeps/lorenz63spantini2022/*.sh``: Lorenz63 experiments (EnFF/EnSF/classical/optuna/reference).

   * ``sweeps/kuramotosivashinsky/*.sh`` and ``sweeps/navierstokesdim256/*.sh``: dataset-specific sweeps.

   * ``sweeps/tune_classical.sh`` and ``sweeps/tune_enkf_letkf.sh``: classical filter sweeps.

   * ``sweeps/rerun.sh``: rerun configurations from existing ``alt_id`` values.

   * ``sweeps/benchmark.sh``: benchmark saved runs by ``alt_id``.

To run these scripts, install `GNU parallel <https://www.gnu.org/software/parallel/>`_.
Once installed, replace ``--eta -j 1`` with ``--dry-run`` in the bash scripts to generate many example commands.


Processing experiment output
============================

We provide Jupyter notebooks in the notebooks directory to process the experiment output:

   * ``notebooks/OptunaHyperparams.ipynb``: Analyze Optuna sweep results and compare best hyperparameters across settings using ``runs.sqlite`` and ``optuna.sqlite``.

   * ``notebooks/AblationSamplingTimeStepCount.ipynb``: Analyze sensitivity to ``filter.sampling_time_step_count`` for EnFF and EnSF runs, and compare with classical filters.

   * ``notebooks/AblationTimeBetweenObs.ipynb``: Analyze sensitivity to ``setting.observe_every_n_time_steps`` for ``Lorenz63Spantini2022``.

   * ``notebooks/BenchmarkTiming.ipynb``: Summarize runtime benchmarks from ``sweeps/benchmark_*.csv`` together with run metadata.
      Use ``sweeps/benchmark.sh`` to generate benchmark outputs.

   * ``notebooks/KalmanRotation2D.ipynb``: Visualize Rotation2D dynamics comparing the Kalman filter and EnFF-F2P.

   * ``notebooks/trajectories_KuramotoSivashinsky.ipynb`` and ``notebooks/trajectories_NavierStokes.ipynb``: Visualize reconstructed trajectories from saved data assimilation runs.
      Use the override ``save_ensemble_stats=true`` to save the reconstructed trajectories.


Running data assimilation experiments in parallel
=================================================

Using `GNU parallel <https://www.gnu.org/software/parallel/>`_, multiple experiments can be run in parallel.

.. code:: bash

   parallel --eta --header : python src/dafm/main.py <override_1>={<param_1>} <override_2>={<param_2>} ... ::: <param_1> <p1value_1> <p1value_2> ... ::: <param_2> <p2value_1> <p2value_2> ...


References
==========

.. [Bao2024] `F. Bao, Z. Zhang, and G. Zhang, "An ensemble score filter for tracking high-dimensional nonlinear dynamical systems," Computer Methods in Applied Mechanics and Engineering, vol. 432, p. 117447, Dec. 2024, doi: 10.1016/j.cma.2024.117447. <https://www.sciencedirect.com/science/article/pii/S0045782524007023>`_
.. [Feng2025] `R. Feng, T. Wu, C. Yu, W. Deng, and P. Hu, "On the Guidance of Flow Matching," Feb. 04, 2025, arXiv: arXiv:2502.02150. doi: 10.48550/arXiv.2502.02150. <http://arxiv.org/abs/2502.02150>`_
