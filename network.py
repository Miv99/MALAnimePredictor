import torch
import user_collector
from sklearn.model_selection import KFold
from collections import defaultdict
import logging
from collections import deque
import os
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np


class Network(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.drop_layer = torch.nn.Dropout(p=0.5)

        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, x):
        y = self.linear2(self.drop_layer(self.relu(self.linear1(x))))
        return y

# Dict of Anime ID : mal_info.DatabaseAnime object
anime_database = user_collector.load_pickle_file('anime_database_test.pickle')
# List of mal_info.User object
users = user_collector.load_pickle_file('users_fit_test.pickle').values()


def purge_anime_list(user):
    """
    user - mal_info.User object

    Removes all non-scored anime in the user's anime list
    Returns size of new anime list
    """
    new_list = {}
    for k in user.anime_list.keys():
        if user.anime_list[k][1] != 0:
            new_list[k] = user.anime_list[k]
    user.anime_list = new_list
    return len(new_list)


# Remove users with < 30 scored anime
users_as_list = [x for x in users if purge_anime_list(x) >= 50]
users_train = [x for x in users_as_list[0:int(len(users_as_list) * 0.8)]]
users_test = [x for x in users_as_list[int(len(users_as_list) * 0.8):len(users_as_list)]]

# ONA, OVA, and Special classified as Special
# Music ignored
types = ['TV', 'Movie', 'Special']

# Find mean value of all features
# Order is same as order of features of x in get_x_list() for simpler code
feature_mean_value = []
s = 0
for user in users_train:
    s += user.mean_score
# user mean score
feature_mean_value.append(s/len(users_train))
s = 0
c = 0
for anime in anime_database.values():
    if anime.mean_score != 0:
        s += anime.mean_score
        c += 1
# anime mean score
feature_mean_value.append(s/c)
# ums of anime with same episode count and similar start date
# both basically the same thing as ums
feature_mean_value.append(feature_mean_value[0])
feature_mean_value.append(feature_mean_value[0])


# Get all genres
genres = set()
percent_dropped_list = []
# sum and count of mean scores by genre and type
s_g = defaultdict(lambda: 0)
c_g = defaultdict(lambda: 0)
s_t = defaultdict(lambda: 0)
c_t = defaultdict(lambda: 0)
for anime in anime_database.values():
    if anime.dropped + anime.watching + anime.completed != 0:
        percent_dropped_list.append(anime.dropped/(anime.dropped + anime.watching + anime.completed))
    for genre in anime.genres:
        genres.add(genre)
        if anime.mean_score != 0:
            s_g[genre] += anime.mean_score
            c_g[genre] += 1
    if anime.mean_score != 0:
        # Treat these as type Special
        if anime.anime_type == 'OVA' or anime.anime_type == 'ONA' or anime.anime_type == 'Special':
            s_t['Special'] += anime.mean_score
            c_t['Special'] += 1
        elif anime.anime_type != 'Music':
            s_t[anime.anime_type] += anime.mean_score
            c_t[anime.anime_type] += 1

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    ret = []
    for i, x in enumerate(data):
        if s[i] < m:
            ret.append(x)
    return ret


# Remove outliers
percent_dropped_list = reject_outliers(percent_dropped_list)
max_percent_dropped = max(percent_dropped_list)
min_percent_dropped = min(percent_dropped_list)
print(min_percent_dropped, max_percent_dropped)
# percent of people that dropped an anime
feature_mean_value.append(np.mean(percent_dropped_list))
# ums of all genres
for genre in genres:
    feature_mean_value.append(s_g[genre]/c_g[genre])
# ums of all types
for anime_type in types:
    feature_mean_value.append(s_t[anime_type]/c_t[anime_type])

combined_roles = {'Art Director' : ['Art Director', 'Director of Photography'],
         'Design' : ['Original Character Design', 'Character Design', 'Color Design', 'Mechanical Design'],
         'Writing' : ['Screenplay', 'Script', 'Series Composition'],
         'Animation Director' : ['Storyboard', 'Assistant Animation Director', 'Animation Director', 'Chief Animation Director'],
         'Animation' : ['Special Effects', 'Key Animation', 'Principle Drawing', '2nd Key Animation', 'Background Art', 'Animation Check', 'Digital Paint', 'Editing', 'In-Between Animation'],
         'Sound' : ['Sound Director', 'Sound Effects'],
         'Producer' : ['Executive Producer', 'Chief Producer', 'Producer', 'Assistant Producer', 'Production Coordination'],
         'Setting' : ['Setting', 'Color Setting'],
         'Director' : ['Assistant Director', 'Episode Director', 'Director'],
         'Creator' : ['Creator', 'Original Creator'],
         'Planning' : ['Planning', 'Layout'],
         'Music' : ['Music'],
         'Main VA' : ['Main'],
         'Supporting VA' : ['Supporting'] }
# To maintain order, just in case
all_combined_role_names = list(combined_roles.keys())


def get_combined_role_name(role_name):
    for combined_name, roles in combined_roles.items():
        if role_name in roles:
            return combined_name
    return None


def get_x_list(user, ids):
    """
    anime_database - Dict of Anime ID : mal_info.DatabaseAnime object
    user - mal_info.User
    ids - array of anime_ids to use

    Mean score features are calculated from anime_train_ids

    Returns 2 lists of length len(anime_test_indices) of tuples of
    (anime_id and tensors), x_list_train and x_list_test

    ums = user mean score
    x = [ums, anime_mean_score, ums_of_anime_with_similar_episode_count, ums_of_anime_with_similar_start_date,
        percent_of_people_that_dropped_the_anime (scaled to be near range [0, 10] before being shifted),
        ums_of_all_genres (0 before shift if genre is not part of the anime's genres),
        ums_of_all_types (0 before shift if type is not the anime's type)]
    """
    shuffle(ids)
    ids_for_ums_calculation = [x for x in ids[0:int(len(ids) * 0.8)]]
    ids_for_training = [x for x in ids[int(len(ids) * 0.8):len(ids)]]

    x_list = []

    ums = 0
    # Doesn't have to be exactly the same; +- 3 is fine
    # Dict with key of episode count and value of ums
    ums_by_episode_count = defaultdict(lambda: 0.)
    # Same^; +- 3 months
    # Dict with key of tuple (month, year) and value of ums
    ums_by_start_date = defaultdict(lambda: 0.)
    ums_by_genre = defaultdict(lambda: 0.)
    ums_by_type = defaultdict(lambda: 0.)
    ums_by_studio = defaultdict(lambda: 0.)

    # Variables for keeping track of sum and count for calculating means
    ums_c = 0
    ums_episode_c = defaultdict(lambda: 0)
    ums_date_c = defaultdict(lambda: 0)
    ums_genres_c = defaultdict(lambda: 0)
    ums_types_c = defaultdict(lambda: 0)
    ums_studios_c = defaultdict(lambda: 0)
    ums_people_c = defaultdict(lambda: defaultdict(lambda: 0))

    ums_s = 0
    ums_episode_s = defaultdict(lambda: 0)
    ums_date_s = defaultdict(lambda: 0)
    ums_genres_s = defaultdict(lambda: 0)
    ums_types_s = defaultdict(lambda: 0)
    ums_studios_s = defaultdict(lambda: 0)
    # Dict with key of people_id and value of
    # dict with keys combined_roles.keys() and value of sum of user scores of all anime in
    # which this people_id was working as this combined role
    ums_people_s = defaultdict(lambda: defaultdict(lambda: 0))

    for anime_id in ids_for_ums_calculation:
        anime = anime_database[anime_id]
        # Skip type music
        if anime.anime_type == 'Music':
            continue

        user_score = user.anime_list[anime_id][1]

        ums_c += 1
        ums_s += user_score

        for i in range(-3, 3):
            ums_episode_c[anime.episodes + i] += 1
            ums_episode_s[anime.episodes + i] += user_score

        for month in range(-3, 3):
            # This happens with anime with start dates that are just the year (which is very few)
            if anime.airing_start_date is None:
                continue
            new_month = anime.airing_start_date.month + month
            new_year = anime.airing_start_date.year
            if new_month <= 0:
                new_year -= 1
                new_month += 12
            elif new_month > 12:
                new_year += 1
                new_month -= 12
            ums_date_c[(new_month, new_year)] += 1
            ums_date_s[(new_month, new_year)] += user_score

        for genre in anime.genres:
            ums_genres_c[genre] += 1
            ums_genres_s[genre] += user_score

        ums_types_c[anime.anime_type] += 1
        ums_types_s[anime.anime_type] += user_score

        for studio in anime.studios:
            ums_studios_c[studio] += 1
            ums_studios_s[studio] += user_score

        for people_id, roles in anime.staff.items():
            for role in roles:
                combined_name = get_combined_role_name(role)
                if combined_name is not None:
                    ums_people_c[people_id][combined_name] += 1
                    ums_people_s[people_id][combined_name] += user_score

    # Calculate all ums stuff
    ums = ums_s / ums_c
    for k in ums_episode_c.keys():
        ums_by_episode_count[k] = ums_episode_s[k] / ums_episode_c[k]
    for k in ums_date_c.keys():
        ums_by_start_date[k] = ums_date_s[k] / ums_date_c[k]
    for k in ums_genres_c.keys():
        ums_by_genre[k] = ums_genres_s[k] / ums_genres_c[k]
    for k in ums_types_c.keys():
        ums_by_type[k] = ums_types_s[k] / ums_types_c[k]
    for k in ums_studios_c.keys():
        ums_by_studio[k] = ums_studios_s[k] / ums_studios_c[k]

    for anime_id in ids_for_training:
        anime = anime_database[anime_id]
        # Skip type music
        if anime.anime_type == 'Music':
            continue

        x = torch.zeros(INPUT_SIZE)
        x[0] = ums
        x[1] = anime_database[anime_id].mean_score
        x[2] = ums_by_episode_count[anime.episodes]
        # This happens with anime with start dates that are just the year (which is very few)
        if anime.airing_start_date is None:
            # Just use the user's mean score
            x[3] = ums
        else:
            x[3] = ums_by_start_date[(anime.airing_start_date.month, anime.airing_start_date.year)]
        # Get as a percent of most watched anime
        # x[4] = (anime.watching + anime.completed)/most_watched
        # x[4] = 5
        percent_dropped = anime.dropped / (anime.watching + anime.completed + anime.dropped)
        # Scale to be near range [0, 10]
        # Will not always be in this range since outliers were removed before finding min and max percent dropped
        x[4] = 10 * (percent_dropped - min_percent_dropped)/(max_percent_dropped - min_percent_dropped)

        studio_score = 0
        if len(anime.studios) != 0:
            for studio in anime.studios:
                studio_score += ums_by_studio[studio]
            studio_score /= len(anime.studios)
        x[5] = studio_score

        i = 6
        for genre in genres:
            if genre in anime.genres:
                x[i] = ums_by_genre[genre]
            else:
                x[i] = 0
            i += 1

        """
        # x[6] is mean of ums of all anime with overlapping genres
        # TODO: increase so that at least [a] number of overlapping genres?
        g_sum = 0
        for genre in anime.genres:
            g_sum += ums_by_genre[genre]
        if len(anime.genres) == 0 or g_sum == 0:
            x[6] = ums
        else:
            x[6] = g_sum / len(anime.genres)
        i = 7
        """
        for anime_type in types:
            if anime_type == 'Special' \
                    and (anime.anime_type == 'OVA' or anime.anime_type == 'ONA' or anime.anime_type == 'Special'):
                x[i] = ums_by_type[anime_type]
            else:
                x[i] = 0
            i += 1

        # considering only related roles
        for combined_role_name in all_combined_role_names:
            role_sum = 0
            role_count = 0
            for people_id, roles in anime.staff.items():
                for role in roles:
                    if get_combined_role_name(role) == combined_role_name:
                        role_sum += ums_people_s[people_id][combined_role_name]
                        role_count += ums_people_c[people_id][combined_role_name]
            if role_count <= 0:
                x[i] = 0
            else:
                x[i] = role_sum / role_count
            i += 1

        # Shift so mean is 0
        for j in range(i):
            if x[j] != 0:
                x[j] -= 5

        x_list.append((anime_id, x))

    return x_list


def get_y(user, anime_id):
    y = torch.zeros(OUTPUT_SIZE)
    y[0] = user.anime_list[anime_id][1]
    return y


# ums = user mean score
# x = [ums, anime_mean_score, ums_of_anime_with_similar_episode_count, ums_of_anime_with_similar_start_date,
#     percent_of_people_that_dropped_the_anime (scaled to be near range [0, 10] before being shifted),
#     ums_of_all_genres (0 before shift if genre is not part of the anime's genres),
#     ums_of_all_types (0 before shift if type is not the anime's type)]
INPUT_SIZE = 6 + len(genres) + len(types) + len(combined_roles)
#INPUT_SIZE = 7 + len(types)
OUTPUT_SIZE = 1

# ----- HYPERPARAMETERS -----
LEARNING_RATE = 1e-3
MOMENTUM = 0
HIDDEN_SIZE = 25
LR_SCHEDULER_GAMMA = 0.3
#LR_SCHEDULER_STEP_SIZE = 1
LR_SCHEDULER_MILESTONES = [2, 4]
WEIGHT_DECAY = 1e-2
# These ones aren't too important
#MAX_ANIME_PER_FOLD = 400
MAX_EPOCH = 200
# For graphing purposes
POINTS_PER_EPOCH = 100
# ----------------------------


def test():
    torch.manual_seed(44)

    criterion = torch.nn.MSELoss(reduction='mean')
    # Stuff that will be saved and can be loaded
    model = Network(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_SCHEDULER_MILESTONES, gamma=LR_SCHEDULER_GAMMA)
    plt_test_x = []
    plt_test_y = []
    plt_train_x = []
    plt_train_y = []
    epoch_train_mean_errors = []
    epoch_test_mean_errors = []
    epoch = 0

    smallest_test = 1e20
    smallest_train = 1e20

    users_train_total = len(users_train)
    users_test_total = len(users_test)

    def save():
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'plt_test_x': plt_test_x,
            'plt_test_y': plt_test_y,
            'plt_train_x': plt_train_x,
            'plt_train_y': plt_train_y,
            'epoch_train_mean_errors': epoch_train_mean_errors,
            'epoch_test_mean_errors': epoch_test_mean_errors
        }, 'model states/' + model_name + '_epoch' + str(epoch) + '.pickle')

    def load(model_name, epoch_to_load):
        nonlocal model
        nonlocal optimizer
        nonlocal scheduler
        nonlocal epoch
        nonlocal plt_test_x
        nonlocal plt_test_y
        nonlocal plt_train_x
        nonlocal plt_train_y
        nonlocal epoch_train_mean_errors
        nonlocal epoch_test_mean_errors

        path = 'model states/' + model_name + '_epoch' + str(epoch_to_load) + '.pickle'

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        plt_test_x = checkpoint['plt_test_x']
        plt_test_y = checkpoint['plt_test_y']
        plt_train_x = checkpoint['plt_train_x']
        plt_train_y = checkpoint['plt_train_y']
        epoch_train_mean_errors = checkpoint['epoch_train_mean_errors']
        epoch_test_mean_errors = checkpoint['epoch_test_mean_errors']

    def show_results_so_far(log):
        print('Epoch test mean errors:')
        if log:
            logging.info('Epoch test mean errors:')
        for i, error in enumerate(epoch_test_mean_errors):
            print('Epoch ' + str(i) + ': ' + str(error))
            if log:
                logging.info('Epoch ' + str(i) + ': ' + str(error))

        print('Epoch train mean errors:')
        if log:
            logging.info('Epoch train mean errors:')
        for i, error in enumerate(epoch_train_mean_errors):
            print('Epoch ' + str(i) + ': ' + str(error))
            if log:
                logging.info('Epoch ' + str(i) + ': ' + str(error))

        print('Smallest test: ' + str(smallest_test))
        print('Smallest train: ' + str(smallest_train))
        if log:
            logging.info('Smallest test: ' + str(smallest_test))
            logging.info('Smallest train: ' + str(smallest_train))

        print("Model's state_dict:")
        if log:
            logging.info('Model\'s state_dict:')
        for param_tensor in model.state_dict():
            print(str(param_tensor) + ': ' + str(model.state_dict()[param_tensor]))
            if log:
                logging.info(str(param_tensor) + ': ' + str(model.state_dict()[param_tensor]))

        plt.plot(plt_test_x, plt_test_y, 'ro')
        plt.plot(plt_train_x, plt_train_y, 'bo')
        plt.show()
        plt.clf()
        plt.gcf().clear()

    if input('Enter 1 to load a model: ') == '1':
        input('')
        model_name = input('Enter model name: ')
        input('')
        epoch = input('Enter epoch: ')
        input('')

        load(model_name, epoch)
    else:
        input('')
        # just to keep track of saved model states
        model_name = input('Enter new model\'s name: ')
        input('')

    logging.basicConfig(filename='logs/' + model_name + '.log',
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO, filemode='a')

    try:
        while epoch < MAX_EPOCH:
            shuffle(users_train)
            scheduler.step()

            user_train_mean_loss_sum = 0
            user_test_mean_loss_sum = 0
            user_train_mean_loss_count = 0
            user_test_mean_loss_count = 0

            i = 0
            epoch_train_sum = 0
            epoch_train_count = 0
            for user in users_train:
                try:
                    keys_as_list = list(user.anime_list.keys())
                    ids = [keys_as_list[x] for x in range(len(keys_as_list))]
                    x_list = get_x_list(user, ids)

                    mean_loss_sum = 0
                    for x in x_list:
                        optimizer.zero_grad()

                        y_pred = model(x[1])
                        y_correct = get_y(user, x[0])

                        loss = criterion(y_pred, y_correct)
                        mean_loss_sum += loss.item()
                        epoch_train_sum += loss.item()
                        loss.backward()
                        optimizer.step()

                        epoch_train_count += 1
                    user_train_mean_loss = mean_loss_sum / len(x_list)

                    user_train_mean_loss_sum += user_train_mean_loss
                    user_train_mean_loss_count += 1

                    if i % int(users_train_total/POINTS_PER_EPOCH) == 0:
                        plt_train_x.append((i + (epoch * users_train_total))/users_train_total)
                        plt_train_y.append(user_train_mean_loss_sum/user_train_mean_loss_count)

                        smallest_train = min(smallest_train, user_train_mean_loss_sum/user_train_mean_loss_count)

                        user_train_mean_loss_sum = 0
                        user_train_mean_loss_count = 0

                    logging.info('epoch ' + str(epoch) + ' user ' + str(i) + '/' + str(users_train_total) + '; avg train = ' + str(user_train_mean_loss))
                    print('epoch ' + str(epoch) + ' user ' + str(i) + '/' + str(users_train_total) + '; avg train = ' + str(user_train_mean_loss))

                    i += 1
                except KeyboardInterrupt:
                    show_results_so_far(False)
                    a = input('Enter 0 to stop training')
                    input('')
                    if a == '0':
                        epoch_train_mean_error = epoch_train_sum / epoch_train_count
                        epoch_train_mean_errors.append(epoch_train_mean_error)
                        raise KeyboardInterrupt
            epoch_train_mean_error = epoch_train_sum / epoch_train_count
            epoch_train_mean_errors.append(epoch_train_mean_error)

            i = 0
            epoch_test_sum = 0
            epoch_test_count = 0
            for user in users_test:
                try:
                    keys_as_list = list(user.anime_list.keys())
                    ids = [keys_as_list[x] for x in range(len(keys_as_list))]
                    x_list = get_x_list(user, ids)

                    mean_loss_sum = 0
                    with torch.no_grad():
                        for x in x_list:
                            y_pred = model(x[1])
                            y_correct = get_y(user, x[0])

                            loss = criterion(y_pred, y_correct)
                            mean_loss_sum += loss.item()
                            epoch_test_sum += loss.item()
                            epoch_test_count += 1
                    user_test_mean_loss = mean_loss_sum / len(x_list)

                    user_test_mean_loss_sum += user_test_mean_loss
                    user_test_mean_loss_count += 1

                    if i % int(users_test_total/POINTS_PER_EPOCH) == 0:
                        plt_test_x.append((i + (epoch * users_test_total))/users_test_total)
                        plt_test_y.append(user_test_mean_loss_sum / user_test_mean_loss_count)

                        smallest_test = min(smallest_test, user_test_mean_loss_sum / user_test_mean_loss_count)

                        user_test_mean_loss_sum = 0
                        user_test_mean_loss_count = 0

                    logging.info(
                        'epoch ' + str(epoch) + ' user ' + str(i) + '/' + str(users_test_total) + '; avg test = ' + str(
                            user_test_mean_loss))
                    print('epoch ' + str(epoch) + ' user ' + str(i) + '/' + str(users_test_total) + '; avg test = ' + str(
                        user_test_mean_loss))

                    i += 1
                except KeyboardInterrupt:
                    show_results_so_far(False)
                    a = input('Enter 0 to stop training')
                    input('')
                    if a == '0':
                        epoch_test_mean_error = epoch_test_sum / epoch_test_count
                        epoch_test_mean_errors.append(epoch_test_mean_error)
                        raise KeyboardInterrupt
            epoch_test_mean_error = epoch_test_sum/epoch_test_count
            epoch_test_mean_errors.append(epoch_test_mean_error)

            epoch += 1
            save()
    except KeyboardInterrupt:
        pass
    finally:
        show_results_so_far(True)
        if input('Enter 1 to save (not recommended if stopped in the middle of an epoch'
                 ' because training resumes from start of epoch): ') == '1':
            save()


if __name__ == '__main__':
    if not os.path.isdir('logs'):
        os.makedirs('logs')

    if not os.path.isdir('model states'):
        os.makedirs('model states')

    test()