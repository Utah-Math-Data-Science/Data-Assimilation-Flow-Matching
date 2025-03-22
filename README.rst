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

      python -c 'import dafm'

#. Edit the ``out_dir`` field of the ``Conf`` class in ``src/conf/conf.py`` to the directory where you want the model training output to be saved.

Supplementary Documentation
===========================

* `Hydra <https://hydra.cc/docs/1.3/intro/>`_: Command-line inferface configuration library for configuring the experiments in this project.
* `PyTorch <https://pytorch.org/docs/2.4/index.html>`_: Library for implementing the models.
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/2.5.0/>`_: Library for handling model training.

Training the models
===================

To train the score matching filter, run

.. code:: bash

   python src/dafm/main.py model=ScoreMatching dataset=DoubleWell model.train_on_initial_predicted_state=true

The override ``model.train_on_initial_predicted_state=true`` has the model trained on the sampled initial predicted states without using any observations.
This follows the implementation from `F. Bao, Z. Zhang, and G. Zhang, "A score-based filter for nonlinear data assimilation," Journal of Computational Physics <https://www.sciencedirect.com/science/article/pii/S002199912400456X>`_.

To train the flow matching filter, run

.. code:: bash

   python src/dafm/main.py model=FlowMatching dataset=DoubleWell model.train_on_initial_predicted_state=true model.loss_expectation_sample_count=1000

The override ``model.loss_expectation_sample_count`` is the number of time-noise pairs ``(t, noise)`` to sample for estimating the expected value in the weighted conditional flow matching loss.

After training, a ``trajectories.parquet`` containing the trajectories of the true system state, the observations, and the predicted system states will be saved to ``<out_dir>/runs/<alt_id>``.
The ``<alt_id>`` is the identifier of the training run, and can be used to visualize the data in ``trajectories.parquet`` using the Jupyter notebooks in ``notebooks``.

To see all the configuration overrides, run

.. code:: bash

   python src/dafm/main.py -c job model=<model> dataset=DoubleWell model.train_on_initial_predicted_state=true

where ``<model>`` is either ``ScoreMatching`` or ``FlowMatching``.
