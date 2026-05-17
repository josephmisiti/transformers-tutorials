from utils import *

from constants import *


download_data(data_folder=DATA_FOLDER)

prepare_data(data_folder=DATA_FOLDER,
             euro_parl=True,
             common_crawl=True,
             news_commentary=True,
             min_length=3,
             max_length=150,
             max_length_ratio=2.,
             retain_case=True)
