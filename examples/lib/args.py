"""Command-line helpers of the example scripts."""

import argparse


def config_first(argv: list[str], parser: argparse.ArgumentParser) -> list[str]:
    """
    Returns argv with the config moved to the front: before it, an unknown --option
    takes the next token as its value.
    """
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("-"):
            return [token, *argv[:i], *argv[i + 1 :]]
        action = parser._option_string_actions.get(token)
        takes_value = action.nargs != 0 if action else token.startswith("--")
        i += 2 if takes_value and "=" not in token else 1
    return argv
