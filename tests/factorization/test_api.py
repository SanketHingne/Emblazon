import os

import numpy as np

import pytest

from emblazon.datasets import movielens
from emblazon.factorization.explicit import ExplicitFactorizationModel
from emblazon.factorization.implicit import ImplicitFactorizationModel


CUDA = bool(os.environ.get('EMBLAZON_CUDA', False))


@pytest.mark.parametrize('model_class', [
    ImplicitFactorizationModel,
    ExplicitFactorizationModel
])
def test_predict_movielens(model_class):

    interactions = movielens.get_movielens_dataset('100K')

    model = model_class(n_iter=1,
                        use_cuda=CUDA)
    model.fit(interactions)

    for user_id in np.random.randint(0, interactions.num_users, size=10):
        user_ids = np.repeat(user_id, interactions.num_items)
        item_ids = np.arange(interactions.num_items)

        uid_predictions = model.predict(user_id)
        iid_predictions = model.predict(user_id, item_ids)
        pair_predictions = model.predict(user_ids, item_ids)

        assert (uid_predictions == iid_predictions).all()
        assert (uid_predictions == pair_predictions).all()
