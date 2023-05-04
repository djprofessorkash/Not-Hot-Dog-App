"""
AUTHOR: Aakash 'Kash' Sudhakar
INTENT: The objective of this file is to produce a basic functional data scraper
        that is able to scrape images of hot dogs and 'not hot dogs' from the 
        ImageNet parent dataset. 
"""

print("Importing libraries...")
import itertools, os, urllib, urllib.request, cv2
# import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
print("Libraries successfully imported.")


class LoadingEngine(object):
    """ Standalone class for loading scraped images as data. """
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

    def load_all_images(self, paths, links):
        for link, path in zip(links, paths):
            if not os.path.exists(path):
                os.makedirs(path)
            print(f"Downloading {link}...")
            self.image_urls = str(urllib.request.urlopen(link).read())

            pool = ThreadPool(processes=32)
            pool.starmap(self.load_single_image, 
                         zip(itertools.repeat(path), 
                             self.image_urls.split("\\n"), 
                             itertools.count(self.DATASIZE)))
            pool.close()
            pool.join()


if __name__ == "__main__":
    # TODO: Refactor loading engine due to major changes to the ImageNet API.
    # links, paths = [
    #     'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01318894', \
    #     'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03405725', \
    #     'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152', \
    #     'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265', \
    #     'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07690019', \
    #     'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07865105', \
    #     'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537' 
    # ], [
    #     "pets",
    #     "furniture",
    #     "people",
    #     "food",
    #     "frankfurter",
    #     "chili-dog",
    #     "hotdog"
    # ]
    # pipeline = LoadingEngine()
    # pipeline.load_all_images(paths=paths, links=links)