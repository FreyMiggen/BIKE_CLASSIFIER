from serpapi import GoogleSearch
from PIL import Image
import os
api_key="<your_api_key>"

def download_and_store(folder,search_words):
    os.makedirs(folder)
    url_list=[]
    for search_word in search_words:
        for i in range(3):
            params = {
              "q": search_word,
              "tbm": "isch",
              "ijn": f"{i}",
              "api_key": api_key
            }

            search = GoogleSearch(params)
            result = search.get_dict()
            images_results= result["images_results"]

            for item in images_results:
                if item['original'] not in url_list:
                    url_list.append(item['original'])

    for count,path in enumerate(url_list):
        url_list[count]=path.split('&')[0]
        
    for count,path in enumerate(url_list,1):
        !wget {path} -O {folder}/{count}.jpg


download_and_store("train/bicycle",["đạp xe quanh bờ hồ","xe đạp ở Hà Nội"])

def clean_data(folder):
    """
    Delete images that are not correctly formatted from folder
    """
    paths=[os.path.join(folder,name) for name in os.listdir(folder)]
    for path in paths:
        try:
            Image.open(path)
        except:
            os.remove(path)