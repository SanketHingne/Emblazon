===================
Model serialization
===================

Saving and loading the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To save a ``emblazon`` model, you can simply use ``torch`` serialization utilities::

  torch.save(emblazon_model, PATH)

and then::

  emblazon_model = torch.load(PATH)
