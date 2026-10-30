# core/loaders/base.py
"""文档加载器的抽象基类。"""
from abc import ABC, abstractmethod

from core.models import Document


class DocumentLoader(ABC):
    """所有加载器的统一接口。

    为什么用 ABC 而不是鸭子类型：
    子类忘记实现 load() 时，会在**实例化时**就报错，
    而不是等到真正调用时才 AttributeError。
    """

    @abstractmethod
    def load(self, path) -> Document:
        """把文件读成 Document。"""

    @abstractmethod
    def supports(self, path) -> bool:
        """这个加载器能否处理该文件。"""