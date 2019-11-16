"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Logger(ABC):

    @abstractmethod
    def record(
        self,
        key,
        value,
    ):
        return NotImplemented
