class User:
    def __init__(self, username='', mean_score=0.0, anime_list={}):
        self.username = username
        self.mean_score = mean_score
        # Dict of anime_id : (watching_status, score_given_by_this_user)
        # 1 = watching, 2 = completed, 4 = dropped, 6 = ptw
        self.anime_list = anime_list

    def __repr__(self):
        return 'mal_info.User{username: ' + str(self.username) + '; mean_score:' + str(self.mean_score)\
               + '; anime list size: ' + str(len(self.anime_list)) + '}'


class Anime:
    def __init__(self, id='', completed=0, watching=0, dropped=0, mean_score=0.0, genres=[],
                 anime_type='', episodes=0, airing_start_date=None):
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
        # Movie/OVA/TV/Special as a string
        self.anime_type = anime_type
        # Total number of episodes
        self.episodes = episodes
        # Anime airing start date as a datetime object
        self.airing_start_date = airing_start_date
        """"
        # Anime rating (PG, R, etc)
        self.rating = rating
        # All important staff IDs as ints
        self.directors = directors
        self.producers = producers
        self.storyboards = storyboards
        self.scripts = scripts
        self.musics = musics
        self.sound_directors = sound_directors
        self.main_voice_actors = main_voice_actors
        self.original_creators = original_creators
        self.animation_directors = animation_directors
        self.episode_directors = episode_directors
        """
    def __repr__(self):
        return '<mal_info.Anime object: ' + str(self.__dict__) + '>'