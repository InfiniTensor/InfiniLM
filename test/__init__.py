# Package marker for the repository test suite -- please do not delete.
#
# This file makes ``test/`` a regular Python package so that dotted-path
# unittest invocations work from the repository root:
#
#     python -m unittest test.agents.test_agents
#
# Without it, ``import test`` resolves to the *standard library* ``test``
# package instead of this directory, and the command above fails with
# "No module named 'test.agents'".
#
# Note: this intentionally shadows the stdlib ``test`` package while running
# the project's own tests; nothing in the test suite imports the stdlib one.
