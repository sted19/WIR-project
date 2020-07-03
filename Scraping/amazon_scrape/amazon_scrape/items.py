# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class AmazonScrapeItem(scrapy.Item):
    # define the fields for your item here like:
    product_asin = scrapy.Field()
    product_category = scrapy.Field()


