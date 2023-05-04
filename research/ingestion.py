"""
AUTHOR: Aakash 'Kash' Sudhakar
INTENT: The objective of this file is to produce a basic functional data scraper
        that is able to scrape images of hot dogs and 'not hot dogs' from the 
        ImageNet parent dataset. 
"""

print("Importing libraries...")
import itertools, os, urllib, cv2
# import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
print("Libraries successfully imported.")


class IngestionEngine(object):
    """ Standalone class for ingesting scraped images as data. """
    def __init__(self):
        self.DATASIZE = 1

    def load_single_image(self, path, link, counter):
        if self.DATASIZE < counter:
            self.DATASIZE = counter + 1
        try:
            full_path_to_image = f"{path}/{str(counter)}.jpg"
            urllib.request.urlretrieve(link,full_path_to_image)
            image = cv2.imread(full_path_to_image)
            if image is not None:
                cv2.imwrite(full_path_to_image, image)
                print(f"Downloaded image `{str(counter)}.jpg`.")
        except Exception as e:
            print(str(e))
            print("Failed to download image.")
            return False

    def load_all_images(paths, links):
        pass

if __name__ == "__main__":
    pipeline = IngestionEngine()