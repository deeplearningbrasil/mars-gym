Install
================================

MARS-Gym is available for Python 3.6. The recommended way to install MARS-Gym is :code:`pip`:

.. code-block:: console

  $ python -m pip install mars-gym

or from the repo:

**Git**

.. code-block:: console


  $ git clone https://gitlab.com/deeplearningbrasil/deep-reco-gym
  $ cd mars-gym
  $ conda env create -f environment.yml
  $ conda activate mars-gym

  $ pip install -e .

Test MARS-Gym installation::

  python -m unittest
