class User:
    def __init__(self, username='', mean_score=0.0, anime_list={}, private_list_or_nonexistent=False):
        self.username = username
        self.mean_score = mean_score
        # Dict of anime_id : (watching_status, score_given_by_this_user)
        # 1 = watching, 2 = completed, 4 = dropped, 6 = ptw
        self.anime_list = anime_list
        # Used to quickly skip users who do not exist or have private anime lists in user_collector
        self.private_list_or_nonexistent = private_list_or_nonexistent

    def __repr__(self):
        return 'mal_info.User{username: ' + str(self.username) + '; mean_score:' + str(self.mean_score)\
               + '; anime list size: ' + str(len(self.anime_list)) + '}'


class Anime:
    def __init__(self, id='', completed=0, watching=0, dropped=0, mean_score=0.0, genres=[],
                 anime_type='', episodes=0, airing_start_date=None, studios=[], staff={}):
        # String
        self.id = id
        # Number of people that have completed the anime
        self.completed = completed
        # Number of people that are watching the anime
        self.watching = watching
        # Number of people that dropped the anime
        self.dropped = dropped
        # Mean score from all users
        self.mean_score = mean_score
        # Genres as name strings
        self.genres = genres
        # Movie/TV/Special/OVA/ONA/Music as a string
        self.anime_type = anime_type
        # Total number of episodes
        self.episodes = episodes
        # Anime airing start date as a datetime object
        self.airing_start_date = airing_start_date
        # Anime studios
        self.studios = studios
        # Dict of staff position string (Director, Producer, etc) : set of people_ids
        # staff['Voice Actor'] contains a dict with keys 'Main' and 'Supporting' and values
        # sets of people_ids of main/supporting voice actors
        self.staff = {}

    def __repr__(self):
        return '<mal_info.Anime object: ' + str(self.__dict__) + '>'