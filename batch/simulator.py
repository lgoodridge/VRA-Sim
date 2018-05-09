from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import shutil
from scipy.sparse.linalg import svds


########################################
#          SIMULATOR SETTINGS
########################################

# Whether to print all log messages to standard out as well
ECHO = False

# Name of the test we're currently running
TEST_NAME = "mytest"

# Directory to save results to, and the log file
SAVE_DIR = ""
LOG_FILE = None

# Total number of films and users
NUM_USERS = 1000 # 6040
NUM_FILMS = 700  # 3883

# Total number of film genres
NUM_GENRES = 2

# Maximum film rating (e.g. 5 uses the rating scale 0-5)
MAX_RATING = 5

# Higher values = stronger user genre preferences
# Use higher values with higher NUM_GENRES values
# Reasonable values are probably b/w 0.1 and 10
USER_POLARIZATION_STRENGTH = 0.5
FILM_POLARIZATION_STRENGTH = 1

# A film is considered "polarized" if its highest
# film genre value is above this threshold
POLARIZED_FILM_THRESHOLD = 0.7

# Percentage of films users will initially watch
INITIAL_VIEWING_RATE = 0.05

# Whether to rescale the initial ratings such that
# mean rating is equal the half the maximum rating
RESCALE_TO_MIDLINE = True

# If True, users will always watch all recommended films
ALWAYS_WATCH = False

# A user's "behavior" determines how a user chooses
# which video(s) to watch at each simulation step
# NAMES is a one-word description of each behavior,
# and DISTRIBUTION determines the ratio of behaviors
# among the created users
BEHAVIOR_PROP_IDX = NUM_GENRES
BEHAVIOR_NAMES = ['follower', 'stubborn', 'mixed']
BEHAVIOR_DISTRIBUTION = [1, 1, 1]

# If False, all users watch recommended films with
# probability equal to their compatibility with it
USE_BEHAVIOR = False

# A film's "quality" determines the mean rating
# of the film, and is used as the baseline which
# user preferences modify to get the final rating
QUALITY_PROP_IDX = NUM_GENRES
QUALITY_MIN = MAX_RATING / 4.0
QUALITY_MAX = 3 * MAX_RATING / 4.0

# If False, the average film rating is used as the
# baseline when calculating new ratings instead
USE_QUALITY = True

# Specify the recommender system to use
AVAILABLE_RECOMMENDER_MODELS = ['svd', 'popularity', 'random']
RECOMMENDER_MODEL = 'svd'

# Number of films to recommend at each step
NUM_RECS = 5

# The number of singular values to compute with SVD
# Larger values give more accurate predictions, but
# take longer to compute (and give longer simulator
# convergence times)
# Reasonable values are probably b/w 10 and 50
K_VAL = 50

# Determines how much a user's compatibility with
# the film affects the final rating. The final
# rating given will be mean_film_rating + x,
# where -CRS <= x <= CRS
COMPATIBILITY_RATING_STRENGTH = 3

# Specify the anti-filter-bubble system to use
AVAILABLE_ANTI_BUBBLE_SYSTEMS = ['none', 'random', 'discovery']
ANTI_BUBBLE_SYSTEM = 'none'

# Determines how much randomness the recommender will
# introduce in order to prevent filter bubbles. The
# final recommendation strength will be the predicted
# rating + x, where -ABS <= x <= ABS
RECOMMENDER_RANDOMNESS = 1.0

# User Discovery Factor. 0 means that user does not want
# to be recommended items of the opposing view.
# 1 means that user really wants to be recommended
# items of the opposing view
# If an item's rating is greater than or equal to 4,
# subtract USER_DISCOVERY_FACTOR * mean_rating from that rating
# If an item's rating is less than or equal to 2, subtract
# add USER_DISCOVERY_FACTOR * mean_rating to that rating
USER_DISCOVERY_FACTOR = 0.5

# Probability of a follower watching a recommended film
FOLLOWER_VIEW_RATE = 0.5

# Minimum compatibility a stubborn user must
# have with a film in order to watch it
STUBBORN_COMPATIBILITY_THRESHOLD = 0.8

# How much watched films are penalized over unwatched
# ones when making recommendations, and when the users
# are choosing which videos to watch respectively
# (Effect is multiplicative: should be b/w 0 and 1)
REWATCH_RECOMMENDATION_MULTIPLIER = 0.5
REWATCH_VIEW_MULTIPLIER = 0.5

# The maximum steps to run the simulation for
MAX_STEPS = 200

# The maximum number of distribution changes allowed
# within a simulator step before its considered "converged"
CHANGE_THRESHOLD = 0


########################################
#           GLOBAL VARIABLES
########################################

users = None
films = None

mean_compatibility = None
max_abs_compatibility = None


########################################
#           CUSTOM EXCEPTIONS
########################################

class TestAlreadyCompletedException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

class OutOfFilmsException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)


########################################
#          SIMULATOR FUNCTIONS
########################################

def log_write(message):
    """
    Writes message to LOG_FILE.
    """
    print(message, file=LOG_FILE)
    if ECHO:
        print(message)

def validate_parameters():
    """
    Ensures the configuration parameters were valid
    and adjusts them as necessary for the simulator.
    """
    global BEHAVIOR_NAMES, BEHAVIOR_DISTRIBUTION, LOG_FILE, SAVE_DIR

    # Validate and standardize the behavior configuration parameters
    if len(BEHAVIOR_NAMES) != len(BEHAVIOR_DISTRIBUTION):
        raise ValueError("BEHAVIOR_NAMES and BEHAVIOR_DISTRIBUTION "
                "must have the same length.")
    BEHAVIOR_NAMES = np.array(BEHAVIOR_NAMES)
    BEHAVIOR_DISTRIBUTION = np.array(BEHAVIOR_DISTRIBUTION) / float(sum(BEHAVIOR_DISTRIBUTION))

    # Validate the model and ABS configuration parameters
    if RECOMMENDER_MODEL not in AVAILABLE_RECOMMENDER_MODELS:
        raise ValueError("RECOMMENDER_MODEL must be a value in AVAILABLE_RECOMMENDER_MODELS.")
    if ANTI_BUBBLE_SYSTEM not in AVAILABLE_ANTI_BUBBLE_SYSTEMS:
        raise ValueError("ANTI_BUBBLE_SYSTEM must be a value in AVAILABLE_ANTI_BUBBLE_SYSTEMS.")

    # Get the save directory using the test name + create it and the log file
    SAVE_DIR = os.path.join("results", TEST_NAME)
    if os.path.exists(SAVE_DIR):
        if os.path.exists(os.path.join(SAVE_DIR, "combined.png")):
            raise TestAlreadyCompletedException(("This test (%s) has "
                    " already been completed. Aborting.") % TEST_NAME)
        else:
            shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    LOG_FILE = open(os.path.join(SAVE_DIR, "log.txt"), "w")

def generate_users(number_of_users):
    """
    Generate a user matrix, where each row of the
    matrix stores the properties of a user:
    [genre1_preference, ..., genreN_preference, behavior]

    The genre preferences of each user sum to 1.
    behavior will be an integer b/w 0 and num_behaviors
    """
    results = []
    for i in range(number_of_users):
        user_prefs = np.random.dirichlet(
                np.ones(NUM_GENRES)/USER_POLARIZATION_STRENGTH, size=1)[0]
        behavior = np.random.choice(np.arange(len(BEHAVIOR_NAMES)),
                p=BEHAVIOR_DISTRIBUTION)
        user_props = np.append(user_prefs, behavior)
        results.append(user_props)
    return np.array(results)

def generate_films(number_of_films):
    """
    Generate a film matrix, where each row of the
    matrix represents the properties of a film:
    [genre1, ..., genreN, quality]

    The genre values of each film sum to 1.
    quality will be b/w QUALITY_MIN and QUALITY_MAX.
    """
    results = []
    for i in range(number_of_films):
        film_genres = np.random.dirichlet(
                np.ones(NUM_GENRES)/FILM_POLARIZATION_STRENGTH, size=1)[0]
        quality = random.uniform(QUALITY_MIN, QUALITY_MAX)
        film_props = np.append(film_genres, quality)
        results.append(film_props)
    return np.array(results)

def divide_film_users_cat(films_or_users):
    """
    Given a film, [genre1, genre2], or a user, [genre1, genre2], and divide
    the film or user into extreme genre1, mild genre1, neutral, mild genre2,
    extreme genre 2.
    Returns cat, which is the category each film belongs to.
    Returns film_num, which is the number of films in each category.
    -2 -> radical genre 1
    -1 -> mild genre 1
    0 -> neutral
    1 -> mild genre 2
    2 -> radical genre 2
    """
    cat = []
    film_num = []
    for film_or_user in films_or_users:
        if film_or_user[0] > film_or_user[1]:
            if abs(film_or_user[0] - film_or_user[1]) <= 0.1:
                cat.append(0)
            elif abs(film_or_user[0] - film_or_user[1]) > 0.1 and abs(film_or_user[0] - film_or_user[1]) <= 0.5:
                cat.append(-1)
            else:
                cat.append(-2)
        else:
            if abs(film_or_user[0] - film_or_user[1]) <= 0.1:
                cat.append(0)
            elif abs(film_or_user[0] - film_or_user[1]) > 0.1 and abs(film_or_user[0] - film_or_user[1]) <= 0.5:
                cat.append(1)
            else:
                cat.append(2)
    extreme1 = np.sum((np.array(cat) == -2).astype(int))
    mild1 = np.sum((np.array(cat) == -1).astype(int))
    neutral = np.sum((np.array(cat) == 0).astype(int))
    mild2 = np.sum((np.array(cat) == 1).astype(int))
    extreme2 = np.sum((np.array(cat) == 2).astype(int))
    film_num.append(extreme1)
    film_num.append(mild1)
    film_num.append(neutral)
    film_num.append(mild2)
    film_num.append(extreme2)
    return cat, film_num

def film_users_cat(film_or_user):
    """
    Given a film, [genre1, genre2], or a user, [genre1, genre2], and divide
    the film or user into extreme genre1, mild genre1, neutral, mild genre2,
    extreme genre 2.
    Return cat, which is the category each film or user belongs to.
    -2 -> radical genre 1
    -1 -> mild genre 1
    0 -> neutral
    1 -> mild genre 2
    2 -> radical genre 2
    """
    cat = 0
    if film_or_user[0] > film_or_user[1]:
        if abs(film_or_user[0] - film_or_user[1]) <= 0.1:
            cat = 0
        elif abs(film_or_user[0] - film_or_user[1]) > 0.1 and abs(film_or_user[0] - film_or_user[1]) <= 0.5:
            cat = -1
        else:
            cat = -2
    else:
        if abs(film_or_user[0] - film_or_user[1]) <= 0.1:
            cat = 0
        elif abs(film_or_user[0] - film_or_user[1]) > 0.1 and abs(film_or_user[0] - film_or_user[1]) <= 0.5:
            cat = 1
        else:
            cat = 2
    return cat

def get_user_film_compatibility(userID, filmID):
    """
    Returns a value between 0 and 1 indicating how much
    a user's preferences aligns with the film's genre(s).
    """
    user_prefs = users[userID][:NUM_GENRES]
    film_genres = films[filmID][:NUM_GENRES]
    return 1 - (np.sum(np.abs(user_prefs - film_genres)) / NUM_GENRES)

def get_user_film_rating(userID, filmID, actual_ratings=None, is_initial_rating=False):
    """
    Returns a value between 0 and MAX_RATING indicating the
    rating the user would give the film upon watching it.

    actual_ratings must be provided when USE_QUALITY and
    is_initial_rating is False, so the current mean rating
    of the film can be calculated.

    is_initial_rating should be set True when generating the
    starting ratings for the film.
    """
    compatibility = get_user_film_compatibility(userID, filmID)

    if USE_QUALITY:
        base_rating = films[filmID][QUALITY_PROP_IDX]
    else:
        if is_initial_rating:
            return round(MAX_RATING * compatibility)
        else:
            film_ratings = actual_ratings[:, filmID]
            base_rating = np.mean(film_ratings[film_ratings.nonzero()])

    # Convert compatibility to a value within [-CRS, +CRS]
    compat_influence = ((compatibility - mean_compatibility)
            / max_abs_compatibility) * COMPATIBILITY_RATING_STRENGTH
    return max(min(round(base_rating + compat_influence), MAX_RATING), 0)

def generate_initial_ratings(users, films, view_rate, rescale_to_midline=False):
    """
    Generates a (num_films, num_users) rating matrix, where
    each row represents the ratings that film has received
    from all users.

    view_rate is the chance a user will view a given film.

    If rescale_to_midline is True, the ratings are scaled
    such that mean of non-zero ratings is MAX_RATING / 2.
    """
    results = []
    for userID in range(users.shape[0]):
        user_ratings = []
        for filmID in range(films.shape[0]):
            ran = random.uniform(0,1)
            if (ran <= view_rate):
                rating = get_user_film_rating(userID, filmID, is_initial_rating=True)
                user_ratings.append(rating)
            else:
                user_ratings.append(0)
        results.append(user_ratings)
    results = np.array(results)
    # Perform rescaling if necessary
    if rescale_to_midline:
        results = results * ((MAX_RATING / 2.0) / (results[results.nonzero()].mean()))
        results = np.clip(np.round(results), 0, MAX_RATING)
    return results

def get_predicted_ratings(actual_ratings):
    """
    Returns a (num_users, num_films) matrix containing
    the predicted ratings each user would each film.
    """
    ratings = actual_ratings

    # If specified, move polarized ratings closer to the median
    # so users can discover content that disagrees with their prefs
    if ANTI_BUBBLE_SYSTEM == 'discovery':
        ratings = np.copy(actual_ratings)
        mask1 = (ratings > 0) & (ratings <= 2)
        mask2 = (ratings >= 4)
        ratings = ratings.astype(float)
        ratings[mask1] = ratings[mask1] + USER_DISCOVERY_FACTOR * np.mean(ratings)
        ratings[mask2] = ratings[mask2] - USER_DISCOVERY_FACTOR * np.mean(ratings)

    if RECOMMENDER_MODEL == 'svd':
        # Only consider non-zero ratings when calculating the mean
        masked_user_ratings = np.ma.masked_equal(ratings, 0)
        user_ratings_mean = masked_user_ratings.mean(axis=1).data
        # Calculate SVD values of demeaned ratings + estimate new ratings
        R_demeaned = ratings - user_ratings_mean.reshape(-1, 1)
        U, sigma, Vt = svds(R_demeaned, k=K_VAL)
        sigma = np.diag(sigma)
        predicted_ratings = np.round(np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1))
        return predicted_ratings

    elif RECOMMENDER_MODEL == 'popularity':
        # IMDB weighted ratings for popularity-based
        # recommendation system: (v/(v+m) * R) + (m/(v+m) * C)
        # v : number of ratings
        # m : minimum votes required
        # R : average ratings of film
        # C : mean ratings of whole report
        v = np.count_nonzero(ratings, axis=0)
        popularity_ratings = pd.DataFrame(v)
        m = popularity_ratings.quantile(0.90)
        m = m.values[0]
        R = pd.DataFrame(ratings).replace(0, np.nan)
        R = R.mean(axis=0)
        C = R.mean()
        popularity_map = (v / (v + m) * R) + (m / (v + m) * C)
        popularity_ratings = pd.DataFrame(ratings).replace(0, np.nan)
        popularity_ratings = popularity_ratings.fillna(popularity_map)
        return popularity_ratings.values

    elif RECOMMENDER_MODEL == 'random':
        random_ratings = pd.DataFrame(ratings).replace(0, np.nan)
        random_idx = random_ratings.isnull()
        random_ratings[random_idx] = np.random.uniform(1, 5,
                (users.shape[0], films.shape[0]))
        return random_ratings.values

def get_recommendations(actual_ratings, predicted_ratings, userID, num_recommendations=NUM_RECS):
    """
    Gets the top num_recommendations film recommendations for
    the provided user, using the rating prediction matrix, and
    returns a tuple containing the recommended film IDs, and
    counts of the recommendations received of each film genre.
    Returns (None, []) if no films can be recommended.
    """
    # Get predicted user ratings + films that were already watched
    predicted_user_ratings = predicted_ratings[userID]
    watched_films = np.apply_along_axis(lambda x: x != 0, 0, actual_ratings[userID])
    # If there are no films that can be recommended, return None
    if np.sum(watched_films) == watched_films.shape[0]:
        return (None, [])
    # Penalize already watched films
    watched_penalty = watched_films * REWATCH_RECOMMENDATION_MULTIPLIER
    watched_penalty[watched_films == 0] = 1.0
    recommendation_matrix = predicted_user_ratings * watched_penalty
    # Add some amount of randomness to the recommendations,
    # according to how much we want to prevent filter bubbles
    if ANTI_BUBBLE_SYSTEM == "random":
        anti_bubble_matrix = (np.random.random(films.shape[0]) - 0.5) * \
                2.0 * RECOMMENDER_RANDOMNESS
        recommendation_matrix += anti_bubble_matrix
    # Get the film recommendations + the genre distribution
    recommended_filmIDs = np.argsort(recommendation_matrix)\
            [-num_recommendations:].tolist()
    film_recs = films[recommended_filmIDs, :]
    genre_counts = [int(round(sum(film_recs[:, genre_idx])))
            for genre_idx in range(NUM_GENRES)]
    return (recommended_filmIDs, genre_counts)

def give_recommendation(userID, filmID, actual_ratings):
    """
    Actually gives a film recommendation to a user, and
    determines whether the user will watch it, based on
    their behavior, and returns the rating if so.
    Returns None if the user does not watch the film.
    """
    behavior_name = BEHAVIOR_NAMES[int(users[userID][BEHAVIOR_PROP_IDX])]
    compatibility = get_user_film_compatibility(userID, filmID)
    ran = random.uniform(0, 1)

    # Follower: watches any of the recommended films with equal weight
    if behavior_name == "follower":
        does_watch = ran <= FOLLOWER_VIEW_RATE
    # Mixed: watches a recommended video with probability
    # equal to their compatability with that film
    elif behavior_name == "mixed":
        does_watch = ran <= compatibility
    # Stubborn: only watches videos above a certain compatibility threshold
    elif behavior_name == "stubborn":
        does_watch = compatibility <= STUBBORN_COMPATIBILITY_THRESHOLD
    else:
        raise ValueError("Programmer Error: Unexpected behavior value '%d'" % behavior)

    # If the user would be rewatching the film, change their
    # mind with some probability, according to the rewatch penalty
    if does_watch and actual_ratings[userID][filmID] != 0:
        ran2 = random.uniform(0, 1)
        does_watch = ran2 <= REWATCH_VIEW_MULTIPLIER

    if does_watch:
        return get_user_film_rating(userID, filmID, actual_ratings)
    else:
        return None

def step_simulation(actual_ratings, predicted_ratings, rec_genre_counts,
                    prev_user_satisfaction_vector):
    """
    Runs one step of the simulation:

    Gives new recommendations to each user, and simulates them
    viewing + rating a random selection of them.

    Returns the new rating matrix, the new distribution matrix of
    recommended film genres, the # of changes in this matrix, a
    vector of all recommended film IDs, and the user satisfaction
    vector.
    """
    new_ratings = actual_ratings.copy()
    new_distribution_matrix = []
    num_distribution_changes = 0
    all_recommended_filmIDs = np.zeros([users.shape[0], NUM_RECS])
    user_satisfaction_vector = np.zeros([users.shape[0]])

    for userID in range(users.shape[0]):
        user = users[userID]
        (recommended_filmIDs, new_genre_counts) = \
                get_recommendations(new_ratings, predicted_ratings, userID)
        if recommended_filmIDs is None:
            raise OutOfFilmsException("Ran out of films to recommend.")
        new_distribution_matrix.append(new_genre_counts)
        all_recommended_filmIDs[userID] = recommended_filmIDs

        # Check for changes in the user's recommendation distribution
        if (new_genre_counts[0] != rec_genre_counts[userID][0]) or \
                (new_genre_counts[1] != rec_genre_counts[userID][1]):
            num_distribution_changes += 1

        # Determine whether the user will watch each
        # recommended film, and assign a rating if so
        user_ratings = []
        for i, filmID in enumerate(recommended_filmIDs):
            rating = None
            if ALWAYS_WATCH:
                rating = get_user_film_rating(userID, filmID, actual_ratings)
            elif USE_BEHAVIOR:
                rating = give_recommendation(userID, filmID, actual_ratings)
            else:
                compatibility = get_user_film_compatibility(userID, filmID)
                has_watched = actual_ratings[userID][filmID] != 0
                ran = random.uniform(0,1)
                if ran <= compatibility * (REWATCH_VIEW_MULTIPLIER if has_watched else 1.0):
                    rating = get_user_film_rating(userID, filmID, actual_ratings)
            if rating is not None:
                new_ratings[userID, filmID] = rating
                user_ratings.append(rating)

        # Determine the user's satisfaction with this step's recommendations
        # If the user did not watch any videos, use the previous satisfaction
        user_satisfaction_vector[userID] = np.mean(user_ratings) \
                if len(user_ratings) > 0 else prev_user_satisfaction_vector[userID]

    return (new_ratings, new_distribution_matrix, num_distribution_changes,
            all_recommended_filmIDs, user_satisfaction_vector)


########################################
#        RUN SIMULATOR FUNCTION
########################################

def run_simulation():
    """
    Runs the simulation, saving the log and plot results.
    """
    global users, films, mean_compatibility, max_abs_compatibility

    # Validate the simulation parameters, then log the start of the test
    validate_parameters()
    log_write("----------------------------------------")
    log_write("STARTING TEST %s:\n" % TEST_NAME)

    # Generate the users and films
    log_write("Generating users and films...")
    users = generate_users(NUM_USERS)
    films = generate_films(NUM_FILMS)

    # Get the mean and max absolute compatibilities
    log_write("Determining the mean and max absolute capabilities...")
    compats = []
    sample_userIDs = random.sample(list(np.arange(NUM_USERS)), min(500, NUM_USERS))
    sample_filmIDs = random.sample(list(np.arange(NUM_FILMS)), min(500, NUM_FILMS))
    for userID in sample_userIDs:
        for filmID in sample_filmIDs:
            compats.append(get_user_film_compatibility(userID, filmID))
    mean_compatibility = 1.0 * sum(compats) / len(compats)
    max_abs_compatibility = max(max(compats), abs(min(compats)))

    # Generate the initial ratings and predictions
    log_write("Generating the initial ratings and predictions...")
    initial_ratings= generate_initial_ratings(users, films,
            INITIAL_VIEWING_RATE, rescale_to_midline=RESCALE_TO_MIDLINE)
    initial_predictions = get_predicted_ratings(initial_ratings)

    # Get the initial distribution matrix
    log_write("Determining the initial distribution matrix...")
    initial_distribution_matrix = []
    for userID in range(users.shape[0]):
        (recommended_filmIDs, type_counts) = get_recommendations(initial_ratings,
                initial_predictions, userID)
        initial_distribution_matrix.append(type_counts)

    # Initialize the rest of the simulator variables
    ratings = initial_ratings.copy()
    predictions = initial_predictions.copy()
    rec_distribution = list(initial_distribution_matrix)
    user_satisfaction = np.zeros(users.shape[0])
    num_changes_over_time = []
    recommended_filmIDs_over_time = []
    user_satisfaction_over_time = []

    # Start stepping through the simulation
    log_write("\nStarting simulation...")
    for step in range(MAX_STEPS):
        predictions = get_predicted_ratings(ratings)
        try:
            ratings, rec_distribution, num_changes, recommended_filmIDs, user_satisfaction = \
                    step_simulation(ratings, predictions, rec_distribution, user_satisfaction)
        except OutOfFilmsException:
            log_write("Ran out of films to recommend.")
            break
        num_changes_over_time.append(num_changes)
        recommended_filmIDs_over_time.append(recommended_filmIDs)
        user_satisfaction_over_time.append(user_satisfaction)
        if step > 0 and num_changes <= CHANGE_THRESHOLD:
            log_write("Convergence!")
            break
        log_write("Step #%d: Num Changes = %d" % (step, num_changes))
        if step == MAX_STEPS-1:
            log_write("Max simulation steps reached.")
    log_write("")
    log_write("Simulation complete. Plotting and saving results...")

    # Plot number of distribution changes over time
    log_write("Saving distribution changes plot...")
    plt.plot(num_changes_over_time)
    plt.xlabel("Step")
    plt.ylabel("# Distribution Changes")
    plt.savefig(os.path.join(SAVE_DIR, "distribution_changes.png"))
    plt.close()

    # Plot the average user satisfaction over time
    log_write("Saving user satisfaction plot...")
    avg_user_satisfaction_over_time = [np.mean(x) for x in user_satisfaction_over_time]
    plt.plot(avg_user_satisfaction_over_time)
    plt.xlabel("Step")
    plt.ylabel("Average User Satisfaction")
    plt.savefig(os.path.join(SAVE_DIR, "user_satisfaction.png"))
    plt.close()

    # Plot the average group satisfaction over time
    log_write("Saving group satisfaction plot...")
    user_cats = [film_users_cat(user) for user in users]
    for cat in range(-2, 3):
        group_satisfaction_over_time = [
            np.mean([x for i,x in enumerate(user_satisfaction) if user_cats[i] == cat])
            for user_satisfaction in user_satisfaction_over_time
        ]
        label = ["Radical Type 1", "Mild Type 1", "Neutral",
                "Mild Type 2", "Radical Type 2"][cat+2]
        plt.plot(group_satisfaction_over_time, label=label)
    plt.xlabel("Step")
    plt.ylabel("Average Group Satisfaction")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "group_satisfaction.png"))
    plt.close()

    # Get the flattened recommended film IDs and genres over time
    flattened_rec_filmIDs_over_time = np.array([x.flatten()
            for x in recommended_filmIDs_over_time])
    flattened_rec_film_genres_over_time = np.array([
            [films[int(filmID)][:NUM_GENRES] for filmID in step_filmIDs]
            for step_filmIDs in flattened_rec_filmIDs_over_time
    ])

    def get_num_polarized_recs(step_rec_film_genres):
        is_polarized = np.apply_along_axis(lambda x: max(x) > POLARIZED_FILM_THRESHOLD,
                1, step_rec_film_genres)
        return sum(is_polarized) / len(step_rec_film_genres)

    # Plot the percentge of recommended films that are polarized over time
    log_write("Saving # polarized recommendations plot...")
    percent_polarized_recs_over_time = [get_num_polarized_recs(x)
            for x in flattened_rec_film_genres_over_time]
    plt.plot(percent_polarized_recs_over_time)
    plt.xlabel("Step")
    plt.ylabel("% Polarized Film Recommendations")
    plt.savefig(os.path.join(SAVE_DIR, "num_polarized_recs.png"))
    plt.close()

    def get_film_polarity(film_genre_vals):
        max_val = max(film_genre_vals)
        return 1.0 * sum([max_val - val for val in film_genre_vals]) / \
                (len(film_genre_vals) - 1)

    def get_avg_film_polarity(step_rec_film_genres):
        return np.mean(np.apply_along_axis(lambda x: get_film_polarity(x),
                1, step_rec_film_genres), axis=0)

    # Plot the average polarity of the film recommendations over time
    log_write("Saving average film polarity plot...")
    avg_rec_film_polarity_over_time = [get_avg_film_polarity(x)
            for x in flattened_rec_film_genres_over_time]
    plt.plot(avg_rec_film_polarity_over_time)
    plt.xlabel("Step")
    plt.ylabel("Average Recommended Film Polarity")
    plt.savefig(os.path.join(SAVE_DIR, "avg_film_polarity.png"))
    plt.close()

    def get_avg_genre_values(step_rec_film_genres):
        return np.mean(step_rec_film_genres, axis=0)

    # Plot the distribution of recommended film genres over time
    log_write("Saving average film genres plot...")
    avg_genre_values_over_time = np.array([get_avg_genre_values(x)
            for x in flattened_rec_film_genres_over_time])
    for genre in range(NUM_GENRES):
        plt.plot(avg_genre_values_over_time[:,genre], label="Genre %d" % (genre+1))
    plt.xlabel("Step")
    plt.ylabel("Average Recommended Film Genre Value")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "avg_film_genre.png"))
    plt.close()

    def plot_filter_bubble(final_rec):
        radical1 = np.int_([0,0,0,0,0])
        mild1 = np.int_([0,0,0,0,0])
        neutral = np.int_([0,0,0,0,0])
        mild2 = np.int_([0,0,0,0,0])
        radical2 = np.int_([0,0,0,0,0])
        for i in range(0,final_rec.shape[0]):
            user = users[i]
            curr_rec = final_rec[i]
            rec_films = films[curr_rec]
            user_cat = film_users_cat(user)
            _,dist = divide_film_users_cat(rec_films)
            if user_cat == -2:
                radical1 += dist
            elif user_cat == -1:
                mild1 += dist
            elif user_cat == 0:
                neutral += dist
            elif user_cat == 1:
                mild2 += dist
            else:
                radical2 += dist
        N = 5
        ind = np.arange(N)  # the x locations for the groups
        width = 0.16        # the width of the bars
        rects1 = plt.bar(ind, radical1, width, color='r')
        rects2 = plt.bar(ind + width, mild1, width, color='y')
        rects3 = plt.bar(ind + 2 * width, neutral, width, color='b')
        rects4 = plt.bar(ind + 3 * width, mild2, width, color='g')
        rects5 = plt.bar(ind + 4 * width, radical2, width, color='c')
        # add some text for labels, title and axes ticks
        plt.ylabel('Number of Films Recommended')
        plt.xticks(ind + width * 2,('Radical \n Type1', 'Mild \n Type1', 'Neutral',
                'Mild \n Type2', 'Radical \n Type2'))
        plt.legend((rects1[0], rects2[0], rects3[0], rects4[0],rects5[0]),
                ('Radical Type1', 'Mild Type1', 'Neutral', 'Mild Type2', 'Radical Type2'))

    # Plot the final filter bubble data
    log_write("Saving filter bubble plot...")
    rec = np.int_(recommended_filmIDs_over_time)[-1]
    plot_filter_bubble(rec)
    plt.savefig(os.path.join(SAVE_DIR, "filter_bubble.png"))
    plt.close()

    # Plot all of the above in a single figure
    log_write("Saving combined plot...")
    plt.figure(figsize=(12, 12))

    plt.subplot(321)
    plt.plot(num_changes_over_time)
    plt.xlabel("Step")
    plt.ylabel("# Distribution Changes")

    plt.subplot(322)
    plot_filter_bubble(rec)

    plt.subplot(323)
    plt.plot(avg_user_satisfaction_over_time)
    plt.xlabel("Step")
    plt.ylabel("Average User Satisfaction")

    plt.subplot(324)
    for cat in range(-2, 3):
        group_satisfaction_over_time = [
            np.mean([x for i,x in enumerate(user_satisfaction) if user_cats[i] == cat])
            for user_satisfaction in user_satisfaction_over_time
        ]
        label = ["Radical Type 1", "Mild Type 1", "Neutral",
                "Mild Type 2", "Radical Type 2"][cat+2]
        plt.plot(group_satisfaction_over_time, label=label)
    plt.xlabel("Step")
    plt.ylabel("Average Group Satisfaction")
    plt.legend()

    plt.subplot(325)
    plt.plot(percent_polarized_recs_over_time)
    plt.xlabel("Step")
    plt.ylabel("% Polarized Film Recommendations")

    plt.subplot(326)
    plt.plot(avg_rec_film_polarity_over_time)
    plt.xlabel("Step")
    plt.ylabel("Average Recommended Film Polarity")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "combined.png"))
    plt.close()

    # Log the end of the test
    log_write("\nFINISHED TEST %s:" % TEST_NAME)
    log_write("----------------------------------------")
    LOG_FILE.close()

