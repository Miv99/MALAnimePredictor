import torch
import user_collector
from sklearn.model_selection import KFold
from datetime import datetime
from collections import defaultdict


class Network(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear2(torch.nn.functional.relu(self.linear1(x)))


# Dict of Anime ID : mal_info.DatabaseAnime object
anime_database = user_collector.load_pickle_file('anime_database_test.pickle')
# List of mal_info.User object
users_fit = user_collector.load_pickle_file('users_fit_test.pickle').values()

BASELINE_DATE = datetime(year=2000, month=1, day=1)


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


def convert_anime_database(anime_database):
    """
    Converts all anime type ONA and OVA to Special
    """


# Remove users with < 30 scored anime
users_fit = [x for x in users_fit if purge_anime_list(x) >= 30]

# Sort by anime list size for faster batch processing
#users_fit.sort(reverse=True, key=lambda x: len(x.anime_list))

# Get all genres
genres = set()
for anime in anime_database.values():
    for genre in anime.genres:
        genres.add(genre)
# ONA, OVA, and Special classified as Special
# Music ignored
types = ['TV', 'Movie', 'Special']


def get_x_list(anime_database, user, anime_train_ids, anime_test_ids):
    """
    anime_database - Dict of Anime ID : mal_info.DatabaseAnime object
    user - mal_info.User
    index - current index

    Returns a list of length len(anime_test_indices) of tuples of
    (anime_id and x tensors), each to be used in separate passes
    ums = user mean score
    x = [ums, anime_mean_score, ums_of_anime_with_similar_episode_count, ums_of_anime_with_similar_start_date,
        total_people_that_completed_the_anime, percent_of_people_that_dropped_the_anime,
        ums_of_all_genres (0 if genre is not part of the anime's genres),
        ums_of_all_types (0 if type is not the anime's type)]
    """
    x_list_train = []
    x_list_test = []

    ums = 0
    # Doesn't have to be exactly the same; +- 3 is fine
    # Dict with key of episode count and value of ums
    ums_by_episode_count = defaultdict(lambda: 0.)
    # Same^; +- 3 months
    # Dict with key of tuple (month, year) and value of ums
    ums_by_start_date = defaultdict(lambda:0.)
    ums_by_genre = defaultdict(lambda: 0.)
    ums_by_type = defaultdict(lambda: 0.)

    # Variables for keeping track of sum and count for calculating means
    ums_c = 0
    ums_episode_c = defaultdict(lambda: 0)
    ums_date_c = defaultdict(lambda: 0)
    ums_genres_c = defaultdict(lambda: 0)
    ums_types_c = defaultdict(lambda: 0)

    ums_s = 0
    ums_episode_s = defaultdict(lambda: 0)
    ums_date_s = defaultdict(lambda: 0)
    ums_genres_s = defaultdict(lambda: 0)
    ums_types_s = defaultdict(lambda: 0)

    for anime_id in anime_train_ids:
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

    # Calculate all ums stuff
    ums = ums_s/ums_c
    for k in ums_episode_c.keys():
        ums_by_episode_count[k] = ums_episode_s[k]/ums_episode_c[k]
    for k in ums_date_c.keys():
        ums_by_start_date[k] = ums_date_s[k]/ums_date_c[k]
    for k in ums_genres_c.keys():
        ums_by_genre[k] = ums_genres_s[k]/ums_genres_c[k]
    for k in ums_types_c.keys():
        ums_by_type[k] = ums_types_s[k]/ums_types_c[k]

    for anime_id in anime_train_ids:
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
        x[4] = anime.watching + anime.completed
        x[5] = anime.dropped/(anime.watching + anime.completed + anime.dropped)
        i = 6
        for genre in genres:
            if genre in anime.genres:
                x[i] = ums_by_genre[genre]
            else:
                x[i] = 0
            i += 1
        for anime_type in types:
            if anime_type == 'Special' \
                    and (anime.anime_type == 'OVA' or anime.anime_type == 'ONA' or anime.anime_type == 'Special'):
                x[i] = ums_by_type[anime_type]
            else:
                x[i] = 0
            i += 1
        x_list_train.append((anime_id, x))

    for anime_id in anime_test_ids:
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
        x[4] = anime.watching + anime.completed
        x[5] = anime.dropped/(anime.watching + anime.completed + anime.dropped)
        i = 6
        for genre in genres:
            if genre in anime.genres:
                x[i] = ums_by_genre[genre]
            else:
                x[i] = 0
            i += 1
        for anime_type in types:
            if anime_type == 'Special' \
                    and (anime.anime_type == 'OVA' or anime.anime_type == 'ONA' or anime.anime_type == 'Special'):
                x[i] = ums_by_type[anime_type]
            else:
                x[i] = 0
            i += 1
        x_list_test.append((anime_id, x))

    return x_list_train, x_list_test


def get_y(user, anime_id):
    y = torch.zeros(OUTPUT_SIZE)
    y[0] = user.anime_list[anime_id][1]
    return y


# ums = user mean score
# x = [ums, anime_mean_score, ums_of_anime_with_similar_episode_count, ums_of_anime_with_similar_start_date,
#  total_people_that_completed_the_anime, percent_of_people_that_dropped_the_anime,
#  ums_of_all_genres (0 if genre is not part of the anime's genres),
#  ums_of_all_types (0 if type is not the anime's type)]
INPUT_SIZE = 6 + len(genres) + len(types)
OUTPUT_SIZE = 1
HIDDEN_SIZE = 40

# Tensors to hold input/output
#x = torch.zeros(BATCH_SIZE, INPUT_SIZE)
#y = torch.zeros(BATCH_SIZE, OUTPUT_SIZE)

device = torch.device('cuda')
model = Network(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE)
#model.cuda(device)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

MAX_ANIME_PER_FOLD = 200

try:
    for epoch in range(1):
        i = 0
        total = len(users_fit)
        for user in users_fit:
            kf = KFold(n_splits=max(2, int(len(user.anime_list)/MAX_ANIME_PER_FOLD)), shuffle=True)
            keys_as_list = list(user.anime_list.keys())
            for train_indices, validation_indices in kf.split(keys_as_list):
                train_ids = [keys_as_list[x] for x in train_indices]
                validation_ids = [keys_as_list[x] for x in validation_indices]
                x_list_train, x_list_test = get_x_list(anime_database, user, train_ids, validation_ids)

                for x in x_list_train:
                    #print('x:', x[1])

                    y_pred = model(x[1])
                    #print('y_pred:', y_pred)
                    y_correct = get_y(user, x[0])
                    #print('y_correct:', y_correct)

                    loss = criterion(y_pred, y_correct)
                    #print('loss:', loss.item())
                    #print('-------------')

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                mean_loss_sum = 0
                for x in x_list_test:
                    with torch.no_grad():
                        y_pred = model(x[1])
                        y_correct = get_y(user, x[0])
                        loss = criterion(y_pred, y_correct)
                        mean_loss_sum += loss.item()

                print(str(i) + '/' + str(total) + ': '  + str(mean_loss_sum/len(x_list_test)))
            i += 1
except KeyboardInterrupt:
    pass

print('Weights: __________________')
for param in model.parameters():
    print(param.data)