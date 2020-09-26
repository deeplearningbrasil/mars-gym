.. MARS documentation master file, created by
   sphinx-quickstart on Sat Jun  6 09:58:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MARS's documentation!
================================

.. note::

   The MARS-Gym is in ‘beta’ and currently under active development. Improvements to the code or documentation are welcome!


MARS-Gym (MArketplace Recommender Systems Gym), a benchmark framework for modeling, training, and evaluating RL-based recommender systems for marketplaces.

Three main components composes the framework:

- Data Engineering Module: A highly customizable module where the consumer can ingest and process a massive amount of data for learning using spark jobs.
- Simulation Module: Holds an extensible module built on top of PyTorch to design learning architectures. It also possesses an OpenAI’s Gym environment that ingests the processed dataset to run a multi-agent system that simulates the targeted marketplace.
- Evaluation Module: Provides a set of distinct perspectives on the agent’s performance. It presents traditional recommendation metrics, off-policy evaluation metrics, and fairness indicators. This component is powered by a user-friendly interface to facilitate the analysis and comparison betweenagents

.. image:: ../images/img2.jpg
  :width: 700

Table of contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   self
   install
   quick_start
   dataviz

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guilde_data_enginnering_module
   guilde_simulation_module
   guilde_evaluation_module

.. toctree::
   :maxdepth: 2
   :caption: MARS

   contrib
   authors
