Contributing
============

Branch strategy
---------------

All changes target ``develop``. ``main`` tracks stable releases only.

.. code-block:: bash

   git clone https://github.com/frapercan/protea-method.git
   cd protea-method
   git checkout develop
   git checkout -b feature/my-feature

   poetry install --with dev

Local checks
------------

Run the full local CI before opening a pull request. There must be zero
"fix CI" follow-up commits:

.. code-block:: bash

   poetry run ruff check .
   poetry run python scripts/check_smells.py --target src
   poetry run mypy src tests
   poetry run pytest --cov=protea_method --cov-fail-under=80

All tests are import-cheap. No GPU or network access is required (the
torch and ESM-2 integration tests skip automatically when their inputs
are absent).

Building the docs
-----------------

.. code-block:: bash

   poetry install --with docs
   poetry run sphinx-build -b html -W docs/source docs/build/html
   # Open docs/build/html/index.html

The build runs with ``-W`` (warnings treated as errors) in CI, so a
broken cross-reference or a missing autodoc target fails the job. Keep
new public modules wired into ``docs/source/reference/index.rst``.

Key constraints
---------------

* **Never use pgvector for KNN search.** FAISS IVFFlat, numpy chunked
  brute-force, or the torch backend are the only allowed retrievers.
* **No runtime dependency** on ``sqlalchemy``, ``fastapi``, or
  ``protea-core``. New dependencies must be optional or justified in the
  pull request description.
* **The LAFA interface is a contract.** ``method_main.py`` must keep its
  generic flags, its 3-column TSV output, and the
  ``/app/data`` + ``/app/output`` bind-mount layout.
* **The feature schema is SemVer-ed.** Coordinate any breaking change
  with ``protea-contracts``; a feature-schema change forces a major bump
  here and re-training of every downstream booster.

Releasing
---------

The container release is operator-driven. Follow
``docker/RELEASE_RUNBOOK.md`` to build, smoke-test, push the image, submit
to FunctionBench, and tag a GitHub release. Keep ``CHANGELOG.md`` current
with a Keep a Changelog ``Unreleased`` section.
