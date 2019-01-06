from datetime import datetime
from mal_scraper import MALScraper
from requests.exceptions import RequestException
from mal_scraper import InvalidUserException
from mal_info import User
import pickle
import logging
from logging.handlers import RotatingFileHandler


def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Max 50mb log file
    handler = RotatingFileHandler('logs/log.log', mode='a', maxBytes=10 * 1024 * 1024)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_pickle_file(pickle_file, default={}):
    try:
        with open(pickle_file, 'rb') as f:
            r = pickle.load(f)
            return r
    except IOError:
        return default


def save(users, anime_database, users_pickle_file_name, anime_database_pickle_file):
    if users is not None:
        with open(users_pickle_file_name + '.pickle', 'wb') as out:
            logging.info('Saving ' + str(len(users)) + ' unique users')
            pickle.dump(users, out, protocol=pickle.HIGHEST_PROTOCOL)
    if anime_database is not None:
        with open(anime_database_pickle_file + '.pickle', 'wb') as out:
            logging.info('Saving ' + str(len(anime_database)) + ' unique anime')
            pickle.dump(anime_database, out, protocol=pickle.HIGHEST_PROTOCOL)


def back_up(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name):
    time_now = str(datetime.now()).replace('.', '_').replace('-', '_').replace(':', '_')
    if users is not None:
        with open('backups/' + users_pickle_file_name + time_now + '.pickle', 'wb') as out:
            pickle.dump(users, out, protocol=pickle.HIGHEST_PROTOCOL)
    if anime_database is not None:
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
    time_now = str(datetime.now()).replace('.', '_').replace('-', '_').replace(':', '_')
    logging.info('-------------------- ' + time_now + ' --------------------------')
    logging.info('usernames_infile = "' + usernames_infile +
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
                logger.info('user_collector: Requesting info for user "' + username + '"')
                autosave_counter += 1
                try:
                    user = scraper.get_user(username)
                    for anime_id in user.anime_list.keys():
                        anime_database[anime_id] = None
                    users[username] = user
                    logger.info('user_collector: Retrieved info for user "' + username + '": ' + str(user))

                    if autosave_counter % autosave_period == 0:
                        save(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)
                        back_up(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)
                except RequestException as e:
                    logger.critical(e)
                except InvalidUserException as e:
                    logger.info(e)
                    users[username] = User(private_list_or_nonexistent=True)
                except Exception as e:
                    logger.fatal(e)
                    users[username] = User(private_list_or_nonexistent=True)
            elif users[username].private_list_or_nonexistent:
                logger.info('user_collector: Skipping user "' + username + '"; user does not exist or has private '
                                                                             'anime list')
            else:
                logger.info('user_collector: Skipping user "' + username + '"; data already exists')

        # Get info for all anime being used by the users
        for anime_id in anime_database.keys():
            print(anime_id)
            if anime_database.get(anime_id) is None:
                logger.info('user_collector: Requesting info for anime id "' + str(anime_id) + '"')
                autosave_counter += 1
                try:
                    anime = scraper.get_anime(str(anime_id))
                    anime_database[anime_id] = anime
                    logger.info('user_collector: Retrieved info for anime id "' + str(anime_id) + '": ' + str(anime))

                    if autosave_counter % autosave_period == 0:
                        save(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)
                        back_up(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)
                except RequestException as e:
                    logger.critical(e)
                except Exception as e:
                    logger.fatal(e)
            else:
                logger.info('user_collector: Skipping anime id "' + str(anime_id) + '"; data already exists')
    except KeyboardInterrupt:
        logger.info('user_collector: Keyboard-interrupted')
    except Exception as e:
        print(e)

    save(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)
    back_up(users, anime_database, users_pickle_file_name, anime_database_pickle_file_name)


def get_staff(anime_database_pickle_file_name, autosave_period):
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
    time_now = str(datetime.now()).replace('.', '_').replace('-', '_').replace(':', '_')
    logging.info('-------------------- ' + time_now + ' --------------------------')
    logging.info('anime_database_pickle_file = "' + anime_database_pickle_file_name + '.pickle')

    # Anime ID : mal_info.Anime object
    anime_database = load_pickle_file(anime_database_pickle_file_name + '.pickle', {})

    scraper = MALScraper(30)
    autosave_counter = 0

    # Encapsulate in try/catch so that process can be keyboard-interrupted and saved at any time
    try:
        # Get info for all anime being used by the users
        for anime_id in anime_database.keys():
            print(anime_id)
            if anime_database.get(anime_id) is None:
                # Request like normal

                logger.info('user_collector: Requesting info for anime id "' + str(anime_id) + '"')
                autosave_counter += 1
                try:
                    anime = scraper.get_anime(str(anime_id))
                    anime_database[anime_id] = anime
                    logger.info('user_collector: Retrieved info for anime id "' + str(anime_id) + '": ' + str(anime))

                    if autosave_counter % autosave_period == 0:
                        back_up(None, anime_database, None, anime_database_pickle_file_name)
                except RequestException as e:
                    logger.critical(e)
                except Exception as e:
                    logger.fatal(e)
            elif not hasattr(anime_database[anime_id], 'staff') or len(anime_database[anime_id].staff) == 0:
                logger.info('user_collector: Requesting staff for anime id "' + str(anime_id) + '"')
                autosave_counter += 1
                try:
                    anime_database[anime_id].staff = scraper.get_staff(anime_id)
                    logger.info('user_collector: Retrieved staff for anime id "' + str(anime_id) + '": '
                                + str(anime_database[anime_id].staff))

                    if autosave_counter % autosave_period == 0:
                        back_up(None, anime_database, None, anime_database_pickle_file_name)
                except RequestException as e:
                    logger.critical(e)
                except Exception as e:
                    logger.fatal(e)
            else:
                logger.info('user_collector: Skipping anime id "' + str(anime_id) + '"; staff data already exists')
    except KeyboardInterrupt:
        logger.info('user_collector: Keyboard-interrupted')
    except Exception as e:
        print(e)

    save(None, anime_database, None, anime_database_pickle_file_name)
    back_up(None, anime_database, None, anime_database_pickle_file_name)


if __name__ == '__main__':
    logger = configure_logger()

    # collect_user_data('users.txt', 'anime_database', 'users_fit', 200)
    get_staff('anime_database', 200)