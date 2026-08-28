# Contributing to MyNNLib

Thanks for considering a contribution.

## Before you start

- Check existing issues before opening a new one.
- For larger changes, open an issue first and describe the intended behavior.
- Keep changes focused. Avoid unrelated formatting or refactors in the same pull request.

## Development setup

~~~bash
git clone https://github.com/Gustavo2820/My_NN_Library.git
cd My_NN_Library
python -m pip install -e ".[dev]"
python -m pytest -q
~~~

## Pull requests

1. Create a branch from the current default branch.
2. Add or update tests for behavior you change.
3. Run python -m pytest -q and make sure it passes.
4. Explain the problem, the solution, and any remaining limitations in the pull-request description.

When changing numerical code, prefer deterministic tests and, where practical, compare analytical gradients with finite differences.

## Reporting bugs

Include the Python and NumPy versions, a minimal reproducible example, the expected result, and the observed result.
