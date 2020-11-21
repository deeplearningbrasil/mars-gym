========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/mars-gym/badge/?style=flat
    :target: https://readthedocs.org/projects/mars-gym
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/deeplearningbrasil/mars-gym.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/deeplearningbrasil/mars-gym

.. |version| image:: https://img.shields.io/pypi/v/mars-gym.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/mars-gym

.. |wheel| image:: https://img.shields.io/pypi/wheel/mars-gym.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/mars-gym

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/mars-gym.svg
    :alt: Supported versions
    :target: https://pypi.org/project/mars-gym

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/mars-gym.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/mars-gym

.. |commits-since| image:: https://img.shields.io/github/commits-since/deeplearningbrasil/mars-gym/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/deeplearningbrasil/mars-gym/compare/v0.1.0...master



.. end-badges

MARS-Gym (MArketplace Recommender Systems Gym), a benchmark framework for modeling, training, and evaluating RL-based recommender systems for marketplaces.

.. figure:: images/img1.jpg
   :alt: MDP

Three main components composes the framework:

- Data Engineering Module: A highly customizable module where the consumer can ingest and process a massive amount of data for learning using spark jobs.
- Simulation Module: Holds an extensible module built on top of PyTorch to design learning architectures. It also possesses an OpenAI’s Gym environment that ingests the processed dataset to run a multi-agent system that simulates the targeted marketplace.
- Evaluation Module: Provides a set of distinct perspectives on the agent’s performance. It presents traditional recommendation metrics, off-policy evaluation metrics, and fairness indicators. This component is powered by a user-friendly interface to facilitate the analysis and comparison betweenagents

.. figure:: images/img2.jpg
   :alt: Framework

   Framework

Dependencies and Requirements
-----------------------------

-  python=3.6.7
-  pandas=0.25.1
-  matplotlib=2.2.2
-  scipy=1.3.1
-  numpy=1.17.0
-  seaborn=0.8.1
-  scikit-learn=0.21.2
-  pytorch=1.2.0
-  tensorboardx=1.6
-  luigi=2.7.5
-  tqdm=4.33
-  requests=2.18.4
-  jupyterlab=1.0.2
-  ipywidgets=7.5.1
-  diskcache=3.0.6
-  pyspark=2.4.3
-  psutil=5.2.2
-  category\_encoders
-  plotly=4.4.1
-  imbalanced-learn==0.4.3
-  torchbearer==0.5.1
-  pytorch-nlp==0.4.1
-  unidecode==1.1.1
-  streamlit==0.52.2
-  gym==0.15.4

Free software: MIT license

Installation
============

::

    pip install mars-gym

You can also install the in-development version with::

    pip install https://github.com/deeplearningbrasil/mars-gym/archive/master.zip


Documentation
=============


https://mars-gym.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox


Usage
-----

Simulate Example
----------------

.. code:: bash


    mars-gym run interaction --project PROJECT \
    --n-factors N_FACTORS --learning-rate LR --optimizer OPTIMIZER \
    --epochs EPOCHS --obs-batch-size OBS_BATCH \
    --batch-size BATCH_SIZE --num-episodes NUM_EP \
    --bandit-policy BANDIT --bandit-policy-params BANDIT_PARAMS

Evaluate Example
----------------

.. code:: bash


    mars-gym evaluate iteraction \
     --model-task-id MODEL_TASK_ID --fairness-columns "[]" \
     --direct-estimator-class DE_CLASS

Evaluation Module
-----------------

.. code:: bash


    mars-gym viz

Cite
----

Please cite the associated paper for this work if you use this code:

::

    @misc{santana2020marsgym,
          title={MARS-Gym: A Gym framework to model, train, and evaluate Recommender Systems for Marketplaces}, 
          author={Marlesson R. O. Santana and Luckeciano C. Melo and Fernando H. F. Camargo and Bruno Brandão and Anderson Soares and Renan M. Oliveira and Sandor Caetano},
          year={2020},
          eprint={2010.07035},
          archivePrefix={arXiv},
          primaryClass={cs.IR}
    }


