import scrapy
from amazon_scrape.items import AmazonScrapeItem


class AmazonproductspiderSpider(scrapy.Spider):
    name = 'AmazonDeals'
    allowed_domains = ['amazon.com']

    #Use working product URL below
    urls_file = open("URLs.txt", "r")
    
    start_urls = []
    
    for line in urls_file:
        url = line.strip()
        start_urls.append(url)

    urls_file.close()



    def parse(self, response):
        items = AmazonScrapeItem()
        asin = response.url.split("/dp/")[1]
        category = response.xpath('//a[@class="a-link-normal a-color-tertiary"]/text()').extract()
        items['product_asin'] = ''.join(asin).strip()
        items['product_category'] = ','.join(map(lambda x: x.strip(), category)).strip()
        yield items
