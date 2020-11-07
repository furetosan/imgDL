import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class ImggetterSpider(CrawlSpider):
    name = 'imggetter'
    # allowed_domains = ['google.com']
    start_urls = ['http://google.com/']

    rules = (
        Rule(LinkExtractor(allow=r''), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        item = {}
        rel = response.xpath('//img/@src').extract()
        item['image_urls'] = [ response.urljoin(rel[0]) ]
        #item['domain_id'] = response.xpath('//input[@id="sid"]/@value').get()
        #item['name'] = response.xpath('//div[@id="name"]').get()
        #item['description'] = response.xpath('//div[@id="description"]').get()
        return item
