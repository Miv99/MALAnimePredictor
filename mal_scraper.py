from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import re
import logging
import json
import time
from mal_info import Anime
from mal_info import User
from datetime import datetime
from collections import deque


class InvalidUserException(Exception):
    pass


# anime_id in %s
ANIME_URL = 'https://myanimelist.net/anime/%s'
ANIME_STATS_URL = 'https://myanimelist.net/anime/%s/_/stats'
ANIME_CHARACTERS_URL = 'https://myanimelist.net/anime/%s/_/characters'
# username and ((page_number - 1) * 300) in %s
ANIME_LIST_URL = 'https://myanimelist.net/animelist/%s/load.json?offset=%s&status=7'

# Max number of cached results
CACHE_MAX_SIZE = 20


class MALScraper:
    def __init__(self, request_delay):
        """
        request_delay - delay in seconds before another request can be made
        if a 429 Too many requests error is encountered
        """
        self.cache = deque()
        self.request_delay = request_delay

    def get_cached_result(self, url):
        for item in self.cache:
            if item[0] == url:
                return item[1]

    def simple_get(self, url):
        """
        Attempts to get the content at `url` by making an HTTP GET request.
        If the content-type of response is some kind of HTML/XML, return the
        text content, otherwise return None.
        """
        logging.info('MALScraper: Requested "' + url + '"')

        # Attempt to get cached result
        cached_result = self.get_cached_result(url)
        if cached_result is not None:
            return cached_result

        try:
            while True:
                with closing(get(url, stream=True)) as resp:
                    # Encountered 429; sleep and try again
                    if resp.content == b'Too Many Requests\n':
                        logging.info('429 Too Many Requests; sleeping for ' + str(self.request_delay) + ' seconds...')
                        time.sleep(self.request_delay)
                        continue

                    # Update cache
                    self.cache.append((url, resp.content))
                    if len(self.cache) > CACHE_MAX_SIZE:
                        self.cache.popleft()

                    return resp.content
        except RequestException as e:
            logging.critical('MALScraper: Error during requests to {0} : {1}'.format(url, str(e)))
            raise e

    def get_mean_score(self, anime_id):
        """
        anime_id - string

        Returns an anime's mean score
        """
        try:
            raw_html = self.simple_get(ANIME_STATS_URL % anime_id)
            html = BeautifulSoup(raw_html, 'html.parser')
            # Sum of scores
            score_sum = 0
            # Sum of votes
            count = 0
            # Enumerate
            score = 10
            for s in html.find_all('small')[-10:]:
                # Number of votes for this score
                c = int(re.findall(r'\d+', str(s))[0])
                count += c
                score_sum += c * score
                score -= 1
            return score_sum/count
        except Exception as e:
            raise e

    def get_anime_viewing_stats(self, anime_id):
        """"
        anime_id - string

        Returns an anime's viewing stats: number watching, completed, on-hold, and dropped, in that order
        """
        try:
            raw_html = self.simple_get(ANIME_STATS_URL % anime_id)
            html = BeautifulSoup(raw_html, 'html.parser')
            result = html.find_all('div', class_='spaceit_pad')

            # Identify lines that contain the stats
            for line in result:
                line = str(line)
                if 'Watching' in line:
                    w_line = line
                elif 'Completed' in line:
                    c_line = line
                elif 'On-Hold' in line:
                    o_line = line
                elif 'Dropped' in line:
                    d_line = line
            watching = int(re.findall(r'\d+', w_line.replace(',', ''))[0])
            completed = int(re.findall(r'\d+', c_line.replace(',', ''))[0])
            on_hold = int(re.findall(r'\d+', o_line.replace(',', ''))[0])
            dropped = int(re.findall(r'\d+', d_line.replace(',', ''))[0])

            return watching, completed, on_hold, dropped
        except Exception as e:
            raise e

    def get_anime_info(self, anime_id):
        """"
        anime_id - string

        Returns an anime's type (TV/Movie/OVA/Special) (string), number of episodes,
        airing start date (datetime), and list of genres (strings)
        """
        try:
            raw_html = self.simple_get(ANIME_URL % anime_id)
            html = BeautifulSoup(raw_html, 'html.parser')

            # Get type
            if html.find('a', href='https://myanimelist.net/topanime.php?type=tv') is not None:
                anime_type = 'TV'
            elif html.find('a', href='https://myanimelist.net/topanime.php?type=special') is not None:
                anime_type = 'Special'
            elif html.find('a', href='https://myanimelist.net/topanime.php?type=ova') is not None:
                anime_type = 'OVA'
            elif html.find('a', href='https://myanimelist.net/topanime.php?type=ona') is not None:
                anime_type = 'ONA'
            elif html.find('a', href='https://myanimelist.net/topanime.php?type=movie') is not None:
                anime_type = 'Movie'
            else:
                anime_type = 'Music'

            # Get number of episodes
            break2 = False
            for text in html.find_all('span', class_='dark_text'):
                if break2:
                    break
                if text.contents[0] == 'Episodes:':
                    for line in text.next_siblings:
                        if len(line) != 0:
                            episodes = line
                            break2 = True
                            break
            try:
                episodes = int(episodes)
            except ValueError:
                # Unknown number of episodes (literally "Unknown")
                episodes = 0

            # Get airing date
            break2 = False
            for text in html.find_all('span', class_='dark_text'):
                if break2:
                    break
                if text.contents[0] == 'Aired:':
                    for line in text.next_siblings:
                        if len(line) != 0:
                            aired = line[3:16]
                            break2 = True
                            break
            # Try different "month, year" and "month day, year"; ignore anime with "year"
            try:
                airing_start_date = datetime.strptime(aired, '%b, %Y\n')
            except ValueError:
                # Check if day < 10 to 0-pad it
                if aired[5] == ',':
                    aired = aired[:4] + '0' + aired[4:-1]
                airing_start_date = datetime.strptime(aired, '%b %d, %Y\n')

            # Get genres
            genres = []
            for text in html.find_all('a', href=re.compile('/anime/genre/')):
                genres.append(str(text.contents[0]))

            # Get studios
            studios = []
            for studios_text in html.find_all('span', class_='dark_text'):
                if studios_text.contents[0] == 'Studios:':
                    for text in studios_text.find_next_siblings('a'):
                        studios.append(str(text.contents[0]))
                    break

            return anime_type, episodes, airing_start_date, genres, studios
        except Exception as e:
            raise e

    def get_anime(self, anime_id):
        """"
        anime_id - string

        Returns a mal_info.Anime object
        """
        try:
            anime_type, episodes, airing_start_date, genres, studios = self.get_anime_info(anime_id)
            watching, completed, on_hold, dropped = self.get_anime_viewing_stats(anime_id)
            mean_score = self.get_mean_score(anime_id)
            return Anime(id=anime_id, completed=completed, watching=watching, dropped=dropped, mean_score=mean_score,
                         genres=genres, anime_type=anime_type, episodes=episodes, airing_start_date=airing_start_date,
                         studios=studios)
        except Exception as e:
            raise e

    def get_anime_list(self, username):
        """"
        username - string

        Returns a user's anime list of dicts of anime info
        """
        try:
            page = 1
            reading = True
            animelist = []
            while reading:
                result = self.simple_get(ANIME_LIST_URL % (username, str((page - 1) * 300)))
                paginated_animelist = json.loads(result)

                if type(paginated_animelist) is dict and paginated_animelist.get('errors') is not None:
                    raise InvalidUserException('User does not exist or anime list is private')

                # Anime lists are capped at 300 anime per page
                if len(paginated_animelist) != 300:
                    reading = False

                animelist.extend(paginated_animelist)
                page += 1
            return animelist
        except Exception as e:
            raise e

    def get_user(self, username):
        """"
        username - string

        Returns a mal_info.User object
        """
        try:
            anime_list = {}
            score_sum = 0
            count = 0
            for anime_info in self.get_anime_list(username):
                anime_list[anime_info['anime_id']] = (anime_info['status'], anime_info['score'])
                # Only count anime with scores
                if anime_info['score'] != 0:
                    score_sum += anime_info['score']
                    count += 1
            if count == 0:
                raise InvalidUserException('User has no scores')
            return User(username=username, mean_score=score_sum/count, anime_list=anime_list)
        except Exception as e:
            raise e