from datetime import datetime
from mal_scraper import MALScraper
from requests.exceptions import RequestException
from mal_scraper import InvalidUserException
from mal_info import User
import pickle
import logging


def load_pickle_file(pickle_file, default={}):
    try:
        with open(pickle_file, 'rb') as f:
            r = pickle.load(f)
            return r
    except IOError:
        return default


def save(users, anime_database, users_pickle_file_name, anime_database_pickle_file):
    with open(users_pickle_file_name + '.pickle', 'wb') as out:
        logging.debug('Saving ' + str(len(users)) + ' unique users')
        pickle.dump(users, out, protocol=pickle.HIGHEST_PROTOCOL)
    with open(anime_database_pickle_file + '.pickle', 'wb') as out:
        logging.debug('Saving ' + str(len(anime_database)) + ' unique anime')
        pickle.dump(anime_database, out, protocol=pickle.HIGHEST_PROTOCOL)


def back_up(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name):
    time_now = str(datetime.now()).replace('.', '_').replace('-', '_').replace(':', '_')
    with open('backups/' + users_pickle_file_name + time_now + '.pickle', 'wb') as out:
        pickle.dump(users, out, protocol=pickle.HIGHEST_PROTOCOL)
    with open('backups/' + anime_database_pickle_file_name + time_now + '.pickle', 'wb') as out:
        pickle.dump(anime_database, out, protocol=pickle.HIGHEST_PROTOCOL)


def collect_user_data(usernames_infile, anime_database_pickle_file_name, users_pickle_file_name, autosave_period):
    """""
    Collect info on all users in usernames_infile and saves the data to pickle files.
    Data is still saved even after a keyboard interrupt.
    
    usernames_infile - newline-separated text file of usernames to be read
    anime_database_pickle_file_name - anime database pickle file path to be read in and overwritten afterwards;
        do not include the .pickle extension;
        dict of anime id as key and mal_info.DatabaseAnime object as value
    users_pickle_file_name - users pickle file path to be read in and overwritten afterwards;
        do not include the .pickle extension
        dict of username as key and mal_info.User object as value
    autosave_period - number of user/anime requests before data is saved and backed-up
    """""
    logging.basicConfig(filename='logs/log.log', filemode='a', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    time_now = str(datetime.now()).replace('.', '_').replace('-', '_').replace(':', '_')
    logging.debug('-------------------- ' + time_now + ' --------------------------')
    logging.debug('usernames_infile = "' + usernames_infile +
                  '"; anime_database_pickle_file = "' + anime_database_pickle_file_name
                  + '.pickle"; users_pickle_file = "' + users_pickle_file_name + '.pickle"')

    with open(usernames_infile) as file:
        usernames = file.read().split('\n')

    # Anime ID : mal_info.Anime object
    anime_database = load_pickle_file(anime_database_pickle_file_name + '.pickle', {})
    # Username : mal_info.User object
    # Empty username corresponds to index where user last left off processing usernames_infile
    users = load_pickle_file(users_pickle_file_name + '.pickle', {})

    scraper = MALScraper(30)
    autosave_counter = 0

    # Encapsulate in try/catch so that process can be keyboard-interrupted and saved at any time
    try:
        # Get info for all users
        for username in usernames:
            print(username)
            if users.get(username) is None:
                logging.debug('user_collector: Requesting info for user "' + username + '"')
                autosave_counter += 1
                try:
                    user = scraper.get_user(username)
                    for anime_id in user.anime_list.keys():
                        anime_database[anime_id] = None
                    users[username] = user
                    logging.debug('user_collector: Retrieved info for user "' + username + '"')

                    if autosave_counter % autosave_period == 0:
                        save(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)
                        back_up(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)
                except RequestException as e:
                    logging.critical(e)
                except InvalidUserException as e:
                    logging.debug(e)
                    users[username] = User(private_list_or_nonexistent=True)
                except Exception as e:
                    logging.fatal(e)
            elif users[username].private_list_or_nonexistent:
                logging.debug('user_collector: Skipping user "' + username + '"; user does not exist or has private '
                                                                             'anime list')
            else:
                logging.debug('user_collector: Skipping user "' + username + '"; data already exists')

        # Get info for all anime being used by the users
        for anime_id in anime_database:
            print(anime_id)
            if anime_database.get(anime_id) is None:
                logging.debug('user_collector: Requesting info for anime id "' + anime_id + '"')
                autosave_counter += 1
                try:
                    anime = scraper.get_anime(anime_id)
                    anime_database[anime_id] = anime
                    logging.debug('user_collector: Retrieved info for anime id "' + anime_id + '"')

                    if autosave_counter % autosave_period == 0:
                        save(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)
                        back_up(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)
                except RequestException as e:
                    logging.critical(e)
                except Exception as e:
                    logging.fatal(e)

            else:
                logging.debug('user_collector: Skipping anime id "' + anime_id + '"; data already exists')
    except KeyboardInterrupt:
        logging.debug('user_collector: Keyboard-interrupted')

    save(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)
    back_up(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)


if __name__ == '__main__':
    collect_user_data('users.txt', 'anime_database', 'users_fit', 200)