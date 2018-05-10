"""
Script for filtering the batch simulator results.

Copies the result directories of all tests matching the provided parameters
    from the 'results' directory into the 'filtered_results' directory.

Example: 'python filter_results.py --model=svd --anti-bubble-system=discovery'
    would copy all results of tests where the SVD recommender and Discovery
    Anti-Filter Bubble system were used.

Run 'python filter_results.py --help' to see a list of all possible options.
"""
import click
import os
import shutil

INPUT_DIR = "results"
OUTPUT_DIR = "filtered_results"

@click.command(help=("Copies folders matching the specified filter from "
        "the '%s' directory to the '%s' directory" % (INPUT_DIR, OUTPUT_DIR)))
@click.option("--model", help="Recommendation model to use",
        type=click.Choice(["", "svd", "popularity", "random"]), default="")
@click.option("--anti-bubble-system", help="Anti filter bubble system to use",
        type=click.Choice(["", "none", "random", "discovery"]), default="")
@click.option("--always-watch", type=click.Choice(["", "True", "False"]),
        help="Whether users always watch all recommended films", default="")
@click.option("--user-polarization", help="User polarization strength",
        type=(float), default=None)
@click.option("--film-polarization", help="Film polarization strength",
        type=(float), default=None)
@click.option("--rewatch-rec-mult", help="Rewatch recommendation multiplier",
        type=(float), default=None)
@click.option("--rewatch-view-mult", help="Rewatch view multiplier",
        type=(float), default=None)
@click.option("--rec-randomness", help="Recommender ABS randomness strength",
        type=(float), default=None)
@click.option("--user-discovery", help="Recommender ABS discovery strength",
        type=(float), default=None)

def _cli_filter_results(model, anti_bubble_system, always_watch,
        user_polarization, film_polarization, rewatch_rec_mult,
        rewatch_view_mult, rec_randomness, user_discovery):

    # Create output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)

    # Create the filter function
    def matches_filter(dirname):
        parts = dirname.split("-")
        if len(parts) != 10:
            return False
        if parts[0] != "test":
            return False
        if user_polarization is not None:
            if parts[1] != ("up%1.1f" % user_polarization):
                return False
        if film_polarization is not None:
            if parts[2] != ("fp%1.1f" % film_polarization):
                return False
        if model != "":
            if parts[3] != model:
                return False
        if anti_bubble_system != "":
            if parts[4] != anti_bubble_system:
                return False
        if always_watch != "":
            if parts[5] != always_watch.lower():
                return False
        if rewatch_rec_mult is not None:
            if parts[6] != ("rrm%1.1f" % rewatch_rec_mult):
                return False
        if rewatch_view_mult is not None:
            if parts[7] != ("rvm%1.1f" % rewatch_view_mult):
                return False
        if rec_randomness is not None:
            if parts[8] != ("rr%1.1f" % rec_randomness):
                return False
        if user_discovery is not None:
            if parts[9] != ("udf%1.1f" % user_discovery):
                return False
        return True

    # Copy all directories matching the filter function
    target_dirs = [x for x in os.listdir(INPUT_DIR) if matches_filter(x)]
    map(lambda x: shutil.copytree(os.path.join(INPUT_DIR, x),
            os.path.join(OUTPUT_DIR, x)), target_dirs)

if __name__ == "__main__":
    _cli_filter_results()
