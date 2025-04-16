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
* `Hydra ORM <https://github.com/reepoi/hydra-orm>`_: My library for saving experiment configurations to an `SQLite <https://sqlite.org/>`_ database.
* `PyTorch <https://pytorch.org/docs/2.4/index.html>`_: Library for implementing the models.
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/2.5.0/>`_: Library for handling model training.

Running the data assilimation algorithms
========================================

Run the command

.. code:: bash

   python src/dafm/main.py dataset=<dataset> model=<model> <other_overrides>...

where:

* ``<dataset>`` is one of:

   * ``DoubleWell``: One-dimensional potential well system from [Bao2024a]_.
   * ``Lorenz63``: Three-dimensional chaotic butterfly attractor system.
   * ``Lorenz96``: :math:`N`-dimensional chaotic system with parameters from [Bao2024a]_.
   * ``Lorenz96BohanEasy``: `Lorenz96` with an easier set of parameters.
   * ``Lorenz96Bohan``: `Lorenz96` with a harder set of parameters.

* ``<model>`` is one of:

   * ``ScoreMatching``: The score matching filter described in [Bao2024a]_.

      * The default parameters are for ``dataset=DoubleWell``.
        Use ``ScoreMatchingLorenz96`` for ``Lorenz96``, and ``ScoreMatchingLorenz96Bohan`` for ``Lorenz96BohanEasy`` and ``Lorenz96Bohan``.

   * ``ScoreMatchingMarginal``: The score matching filter described in [Bao2024b]_.

      * **Implementation coming soon.**

   * ``FlowMatching``: Our flow matching filter that requires training.

      * The default parameters are for ``dataset=DoubleWell``.
        Use ``FlowMatchingLorenz96`` for ``Lorenz96``, and ``FlowMatchingLorenz96Bohan`` for ``Lorenz96BohanEasy`` and ``Lorenz96Bohan``.

   * ``FlowMatchingMarginal``: Our flow matching filter that does **not** require training.

      * The default parameters are for ``dataset=DoubleWell``.
        Use ``FlowMatchingLorenz96Bohan`` for ``Lorenz96BohanEasy`` and ``Lorenz96Bohan``.

* ``<other_overrides>...``: Other overrides for the model.
   Add the flag ``-c job`` to the Python command see what can be overridden from the command line.
   Some useful overrides include:

   * Changing the diffusion path:

      .. code:: bash

         model/diffusion_path=VarianceExploding

   * Using particle noise perturbation:

      .. code:: bash

         model.use_state_perturbation=true model.state_perturbation_std=0.5

   * Using particle inflation:

      .. code:: bash

         model/inflation_scale=ConstantScale model.inflation_scale.constant=1.01

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
