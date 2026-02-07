""" Fetch data from FRED """
import pandas as pd
import numpy as np
import logging
from fredapi import Fred
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

