import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
process = CrawlerProcess(settings=get_project_settings())
memes_file = open('meme_list.txt','r')
memes = memes_file.read()
memes_file.close()
ms = memes.split('\n')
for m in ms:
    ss = 'meme '+m
    process.crawl('imggetter',start_urls=[
        # "https://google.com/search?q={name}".format(name=ss),
        # "https://bing.com/search?q={name}".format(name=ss),
        # "https://duckduckgo.com/?q={name}".format(name=ss),
        "http://9gag.com/",
      ])
process.start()

