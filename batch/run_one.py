"""
Runs a single simulation.
"""
from __future__ import print_function
import simulator
import sys

# Set the simulation parameters
user_polarization = 0.5
film_polarization = 0.5
model = 'svd'
always_watch = False
rewatch_rec_mult = 0.5
rewatch_view_mult = 0.5
anti_bubble_sys = 'none'
recommender_randomness = 2.5
user_discovery_factor = 0.5

# Announce the start of the simulation
test_name = "test-up%1.1f-fp%1.1f-%s-%s-%s-rrm%1.1f-rvm%1.1f-rr%1.1f-udf%1.1f" % (user_polarization, film_polarization, model, anti_bubble_sys, str(always_watch).lower(), rewatch_rec_mult, rewatch_view_mult, recommender_randomness, user_discovery_factor)
print("Running %s... " % test_name, end="")
sys.stdout.flush()

# Actually set the simulator module's parameters
simulator.TEST_NAME = test_name
simulator.MAX_STEPS = 100
simulator.USER_POLARIZATION_STRENGTH = user_polarization
simulator.FILM_POLARIZATION_STRENGTH = film_polarization
simulator.RECOMMENDER_MODEL = model
simulator.ALWAYS_WATCH = always_watch
simulator.REWATCH_RECOMMENDATION_MULTIPLIER = rewatch_rec_mult
simulator.REWARCH_VIEW_MULTIPLIER = rewatch_view_mult
simulator.ANTI_BUBBLE_SYSTEM = anti_bubble_sys
simulator.RECOMMENDER_RANDOMNESS = recommender_randomness
simulator.USER_DISCOVERY_FACTOR = user_discovery_factor

# Run the simulation and print the result
try:
    simulator.run_simulation()
    print("Succeeded")
except simulator.TestAlreadyCompletedException as e:
    print("Previously Completed")
except Exception as e:
    print("FAILED!\n%s" % str(e))
