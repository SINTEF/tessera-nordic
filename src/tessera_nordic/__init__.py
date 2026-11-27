"""tessera_nordic: cluster TESSERA satellite embeddings over Norway and Sweden."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tessera-nordic")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"
