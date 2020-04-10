"""Methods for training DeepSpeech language model."""
import logging
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

import networkx as nx
import rhasspynlu

_LOGGER = logging.getLogger("rhasspyasr_deepspeech_hermes")


def train(
    graph: nx.DiGraph,
    language_model: typing.Union[str, Path],
    trie_path: typing.Union[str, Path],
    alphabet_path: typing.Union[str, Path],
    balance_counts: bool = True,
):
    """Re-generates language model and trie from intent graph"""
    with tempfile.NamedTemporaryFile(mode="w+") as arpa_file:
        # 1. Create language model
        _LOGGER.debug("Converting to ARPA language model")
        rhasspynlu.arpa_lm.graph_to_arpa(graph, arpa_file.name)

        arpa_file.seek(0)

        with tempfile.NamedTemporaryFile(mode="w+") as lm_file:
            # 2. Convert to binary language model
            arpa_to_binary(arpa_file.name, lm_file.name)

            lm_file.seek(0)

            with tempfile.NamedTemporaryFile(mode="w+") as trie_file:
                # 3. Generate trie
                make_trie(alphabet_path, lm_file.name, trie_file.name)

                # Copy over actual files
                lm_file.seek(0)
                shutil.copy(lm_file.name, language_model)
                _LOGGER.debug("Wrote binary language model to %s", language_model)

                trie_file.seek(0)
                shutil.copy(trie_file.name, trie_path)
                _LOGGER.debug("Wrote trie to %s", trie_path)


def arpa_to_binary(
    arpa_path: typing.Union[str, Path], binary_lm_path: typing.Union[str, Path]
):
    """Convert ARPA language model to binary format using kenlm."""
    binary_command = ["build_binary", "-T", "-s", str(arpa_path), str(binary_lm_path)]
    _LOGGER.debug(binary_command)
    subprocess.check_call(binary_command)


def make_trie(
    alphabet_path: typing.Union[str, Path],
    binary_lm_path: typing.Union[str, Path],
    trie_path: typing.Union[str, Path],
):
    """Generate trie using Mozilla native-client tool."""
    trie_command = [
        "generate_trie",
        str(alphabet_path),
        str(binary_lm_path),
        str(trie_path),
    ]

    _LOGGER.debug(trie_command)
    subprocess.check_call(trie_command)
