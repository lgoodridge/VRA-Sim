"""
Runs several simulations at once.
"""
from __future__ import print_function
import datetime
import os
import simulator
import sys

# Create the master log file for this batch
log_name = "log-" + datetime.datetime.now().strftime("%m-%d-%H%M%S.txt")
master_log = open(os.path.join("master_logs", log_name), "w")
print("-- START_LOG --", file=master_log)

test_idx = 0
NUM_TESTS = 252

def run_test(user_polarization, film_polarization, model, always_watch,
        rewatch_rec_mult, rewatch_view_mult, anti_bubble_sys,
        recommender_randomness, user_discovery_factor):
    """
    Runs a test with the provided simulator parameters.
    """
    test_name = "test-up%1.1f-fp%1.1f-%s-%s-%s-rrm%1.1f-rvm%1.1f-rr%1.1f-udf%1.1f" % (user_polarization, film_polarization, model, anti_bubble_sys, str(always_watch).lower(), rewatch_rec_mult, rewatch_view_mult, recommender_randomness, user_discovery_factor)
    print("Running %s..." % test_name, file=master_log)
    print("Running (%d of %d) %s... " % (test_idx, NUM_TESTS, test_name), end="")
    master_log.flush()
    sys.stdout.flush()

    reload(simulator)
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

    try:
        simulator.run_simulation()
        print("Succeeded")
    except simulator.TestAlreadyCompletedException as e:
        print("Previously Completed", file=master_log)
        print("Previously Completed")
    except Exception as e:
        print("FAILED: %s" % str(e), file=master_log)
        print("FAILED!\n%s" % str(e))

# Run a simulation for all combinations of the parameters we want to test
for user_polarization in [0.5]:
    for film_polarization in [0.5]:
        for model in ['svd', 'popularity', 'random']:
            for always_watch in [True, False]:
                for rewatch_rec_mult in [0.5, 1.0, 0.0]:
                    possible_rvms = [0.5, 1.0, 0.0] if not always_watch else [1.0]
                    for rewatch_view_mult in possible_rvms:
                        for anti_bubble_sys in ['none', 'random', 'discovery']:
                            possible_rrs = [2.5, 1.0, 10.0] if anti_bubble_sys == 'random' else [0.0]
                            possible_udfs = [0.25, 0.5, 1.0] if anti_bubble_sys == 'discovery' else [0.0]
                            for recommender_randomness in possible_rrs:
                                for user_discovery_factor in possible_udfs:
                                    test_idx += 1
                                    run_test(user_polarization, film_polarization, model, always_watch, rewatch_rec_mult, rewatch_view_mult, anti_bubble_sys, recommender_randomness, user_discovery_factor)

# Cleanup and close the log
print("-- END LOG --", file=master_log)
master_log.close()
