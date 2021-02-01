from typing import Union  # NOQA

# from optuna.storages.base import BaseStorage
# from optuna.storages.in_memory import InMemoryStorage
from .base import BaseStorage
from .in_memory import InMemoryStorage

def get_storage(storage):
    # type: (Union[None, str, BaseStorage]) -> BaseStorage
    if storage is None:
        return InMemoryStorage()
    else:
        return storage
