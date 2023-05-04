"""
AUTHOR: Aakash 'Kash' Sudhakar
INTENT: The objective of this file is to produce a basic functional data scraper
        that is able to scrape images of hot dogs and 'not hot dogs' from the 
        ImageNet parent dataset. 
"""

class IngestionEngine(object):
    """ Standalone class for ingesting scraped images as data. """
    def __init__(self):
        # print("Importing libraries...")
        import itertools, os, urllib, cv2
        import numpy as np
        from multiprocessing.dummy import Pool as ThreadPool
        # print("Libraries successfully imported.")

if __name__ == "__main__":
    pipeline = IngestionEngine()