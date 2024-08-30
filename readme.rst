.. image:: docs/_static/img/emblazon.png

---------------------------------------------------------------------

.. inclusion-marker-do-not-remove

.. image:: https://travis-ci.org/maciejkula/emblazon.svg?branch=master
   :target: https://travis-ci.org/maciejkula/emblazon

.. image:: https://ci.appveyor.com/api/projects/status/jq5e76a7a08ra2ji/branch/master?svg=true
   :target: https://ci.appveyor.com/project/maciejkula/emblazon/branch/master

.. image:: https://badges.gitter.im/gitterHQ/gitter.png
   :target: https://gitter.im/emblazon-recommendations/Lobby

.. image:: https://anaconda.org/maciejkula/emblazon/badges/version.svg
   :target: https://anaconda.org/maciejkula/emblazon

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://maciejkula.github.io/emblazon/

.. image:: https://img.shields.io/badge/progress%20tracker-trello-brightgreen.svg
   :target: https://trello.com/b/G5iFgS1W/emblazon

|

Emblazon uses `PyTorch <http://pytorch.org/>`_ to build both deep and shallow
recommender models. By providing both a slew of building blocks for loss functions
(various pointwise and pairwise ranking losses), representations (shallow
factorization representations, deep sequence models), and utilities for fetching
(or generating) recommendation datasets, it aims to be a tool for rapid exploration
and prototyping of new recommender models.

See the full `documentation <https://maciejkula.github.io/emblazon/>`_ for details.

Installation
~~~~~~~~~~~~

.. code-block:: python

   conda install -c maciejkula -c pytorch emblazon


Usage
~~~~~

Factorization models
====================

To fit an explicit feedback model on the MovieLens dataset:

.. code-block:: python

    from emblazon.cross_validation import random_train_test_split
    from emblazon.datasets.movielens import get_movielens_dataset
    from emblazon.evaluation import rmse_score
    from emblazon.factorization.explicit import ExplicitFactorizationModel

    dataset = get_movielens_dataset(variant='100K')

    train, test = random_train_test_split(dataset)

    model = ExplicitFactorizationModel(n_iter=1)
    model.fit(train)

    rmse = rmse_score(model, test)



To fit an implicit ranking model with a BPR pairwise loss on the MovieLens dataset:

.. code-block:: python

    from emblazon.cross_validation import random_train_test_split
    from emblazon.datasets.movielens import get_movielens_dataset
    from emblazon.evaluation import mrr_score
    from emblazon.factorization.implicit import ImplicitFactorizationModel

    dataset = get_movielens_dataset(variant='100K')

    train, test = random_train_test_split(dataset)

    model = ImplicitFactorizationModel(n_iter=3,
                                       loss='bpr')
    model.fit(train)

    mrr = mrr_score(model, test)




Sequential models
=================

Recommendations can be seen as a sequence prediction task: given the items a user
has interacted with in the past, what will be the next item they will interact
with? Emblazon provides a range of models and utilities for fitting next item
recommendation models, including

- pooling models, as in `YouTube recommendations <https://pdfs.semanticscholar.org/bcdb/4da4a05f0e7bc17d1600f3a91a338cd7ffd3.pdf>`_,
- LSTM models, as in `Session-based recommendations... <https://arxiv.org/pdf/1511.06939>`_, and
- causal convolution models, as in `WaveNet <https://arxiv.org/pdf/1609.03499>`_.

.. code-block:: python

    from emblazon.cross_validation import user_based_train_test_split
    from emblazon.datasets.synthetic import generate_sequential
    from emblazon.evaluation import sequence_mrr_score
    from emblazon.sequence.implicit import ImplicitSequenceModel

    dataset = generate_sequential(num_users=100,
                                  num_items=1000,
                                  num_interactions=10000,
                                  concentration_parameter=0.01,
                                  order=3)

    train, test = user_based_train_test_split(dataset)

    train = train.to_sequence()
    test = test.to_sequence()

    model = ImplicitSequenceModel(n_iter=3,
                                  representation='cnn',
                                  loss='bpr')
    model.fit(train)

    mrr = sequence_mrr_score(model, test)


  

Datasets
========

Emblazon offers a slew of popular datasets, including Movielens 100K, 1M, 10M, and 20M.
It also incorporates utilities for creating synthetic datasets. For example, `generate_sequential`
generates a Markov-chain-derived interaction dataset, where the next item a user chooses is
a function of their previous interactions:

.. code-block:: python

    from emblazon.datasets.synthetic import generate_sequential

    # Concentration parameter governs how predictable the chain is;
    # order determins the order of the Markov chain.
    dataset = generate_sequential(num_users=100,
                                  num_items=1000,
                                  num_interactions=10000,
                                  concentration_parameter=0.01,
                                  order=3)




Examples
~~~~~~~~

1. `Rating prediction on the Movielens dataset <https://github.com/maciejkula/emblazon/tree/master/examples/movielens_explicit>`_.
2. `Using causal convolutions for sequence recommendations <https://github.com/maciejkula/emblazon/tree/master/examples/movielens_sequence>`_.
3. `Bloom embedding layers <https://github.com/maciejkula/emblazon/tree/master/examples/bloom_embeddings>`_.


How to cite
~~~~~~~~~~~

Please cite Emblazon if it helps your research. You can use the following BibTeX entry:

.. code-block::

   @misc{kula2017emblazon,
     title={Emblazon},
     author={Kula, Maciej},
     year={2017},
     publisher={GitHub},
     howpublished={\url{https://github.com/maciejkula/emblazon}},
   }


Contributing
~~~~~~~~~~~~

emblazon is meant to be extensible: pull requests are welcome. Development progress is tracked on `Trello <https://trello.com/b/G5iFgS1W/emblazon>`_: have a look at the outstanding tickets to get an idea of what would be a useful contribution.

We accept implementations of new recommendation models into the Emblazon model zoo: if you've just published a paper describing your new model, or have an implementation of a model from the literature, make a PR!
