"""Structural tests for the two lookup tables at the core of the agent.

These cover ``GetState.getState`` and ``ActionSelection.getPhaseDuration``,
the only two modules in ``ReinforcementLearningPack`` that are pure functions
with no Aimsun dependency. Both are syntax-compatible with Python 3, so these
tests run anywhere -- no simulator and no ``_AAPI`` extension needed.

The properties asserted here are the design invariants of the MDP, not
implementation details:

* the state encoder is a bijection between the 24 orderings of four approach
  queues and the 24 state indices;
* every action leaves the same total green time in the cycle, so the agent can
  only redistribute green, never create or destroy it.

Run with::

    python -m unittest discover -s tests -v
"""

import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ReinforcementLearningPack import ActionSelection, GetState  # noqa: E402

NUMBER_OF_STATES = 24
NUMBER_OF_ACTIONS = 19
TOTAL_GREEN_PER_CYCLE = 92


def queues_for_ordering(ordering):
    """Build four distinct queue lengths realising ``ordering``.

    ``ordering`` is a permutation of the approach indices 0..3, longest queue
    first. The returned values are strictly decreasing in that order, so the
    encoder sees an unambiguous ranking.
    """
    queues = [0] * 4
    for rank, approach in enumerate(ordering):
        queues[approach] = 40 - rank * 10
    return queues


class GetStateTest(unittest.TestCase):
    """``getState`` must rank the four approach queues into a state index."""

    def test_every_ordering_maps_to_a_distinct_state(self):
        """The 24 orderings and the 24 states are in one-to-one correspondence.

        This is what makes ``numberOfState = 24`` in ``Main.py`` correct. A
        transposition typo in one branch silently aliases two orderings onto
        the same state and strands another state as unreachable, which is
        exactly the failure this test exists to catch.
        """
        orderings = list(itertools.permutations(range(4)))
        self.assertEqual(len(orderings), NUMBER_OF_STATES)

        states = [GetState.getState(queues_for_ordering(o)) for o in orderings]

        self.assertEqual(
            sorted(states),
            list(range(NUMBER_OF_STATES)),
            "each of the 24 queue orderings must map to its own state index",
        )

    def test_state_is_in_range(self):
        """No input may produce an index outside the Q-table's row count."""
        for ordering in itertools.permutations(range(4)):
            state = GetState.getState(queues_for_ordering(ordering))
            self.assertIn(state, range(NUMBER_OF_STATES))

    def test_all_queues_equal_is_accepted(self):
        """Ties are legal: the comparisons are ``>=``, so a tie hits branch 0."""
        self.assertEqual(GetState.getState([0, 0, 0, 0]), 0)


class GetPhaseDurationTest(unittest.TestCase):
    """``getPhaseDuration`` maps an action index to four green durations."""

    def test_every_action_preserves_total_green_time(self):
        """Actions redistribute green time; they never change how much there is.

        ``Main.py`` runs a fixed 100 s control cycle and writes green times to
        the four odd-numbered phases only. Holding the sum constant leaves the
        remaining 8 s for the untouched even-numbered inter-green phases, so
        the cycle length stays fixed no matter what the agent chooses.
        """
        for action in range(NUMBER_OF_ACTIONS):
            durations = ActionSelection.getPhaseDuration(action)
            self.assertEqual(len(durations), 4)
            self.assertEqual(
                sum(durations),
                TOTAL_GREEN_PER_CYCLE,
                "action %d changes the total green time" % action,
            )

    def test_actions_are_distinct(self):
        """No two action indices may encode the same split."""
        splits = [tuple(ActionSelection.getPhaseDuration(a)) for a in range(NUMBER_OF_ACTIONS)]
        self.assertEqual(len(set(splits)), NUMBER_OF_ACTIONS)

    def test_action_set_is_the_three_expected_families(self):
        """The 19 actions are 6 + 12 + 1 rearrangements of three duration sets."""
        families = {}
        for action in range(NUMBER_OF_ACTIONS):
            shape = tuple(sorted(ActionSelection.getPhaseDuration(action), reverse=True))
            families[shape] = families.get(shape, 0) + 1

        self.assertEqual(
            families,
            {
                (33, 33, 13, 13): 6,   # two long, two short
                (33, 23, 23, 13): 12,  # one long, one short, two even
                (23, 23, 23, 23): 1,   # the uniform split, the default branch
            },
        )

    def test_out_of_range_action_falls_back_to_the_uniform_split(self):
        """The function's default branch is the even split, as its comment says."""
        self.assertEqual(ActionSelection.getPhaseDuration(NUMBER_OF_ACTIONS), [23, 23, 23, 23])


if __name__ == "__main__":
    unittest.main()
