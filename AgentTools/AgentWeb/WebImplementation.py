from .WebInterface import WebInterface
import requests
from bs4 import BeautifulSoup
import time
import platform
from readability import Document
from pyppeteer import launch
from pyppeteer_stealth import stealth  # Import the stealth plugin

class WebImplementation(WebInterface):
    SERP_API_KEY = 'b4d8added80a7a3796b5e1c8cd010d883a3fcbd8775260847ac7dc4e3100d0dc'

    async def _get_browser(self):
        browser = await launch(headless=True, args=['--no-sandbox'])
        await stealth(browser)  # Apply the stealth plugin to the browser
        return browser

    async def _get_page(self, url):
        browser = await self._get_browser()
        page = await browser.newPage()
        await page.goto(url)
        return page

    def search(self, query, category=None):
        endpoint = 'https://serpapi.com/search'
        params = {
            'q': query,
            'api_key': self.SERP_API_KEY
        }
        if category == 'news':
            params['tbm'] = 'nws'
        response = requests.get(endpoint, params=params)
        return response.json()

    def category_search(self, category):
        return this.search('', category)

    async def load_page(self, url):
        page = await self._get_page(url)
        page_source = await page.content()
        await page.browser().close()
        return page_source

    async def check_expandable_content(self, url):
        page = await self._get_page(url)
        expandable_contents = []
        iframes = await page.querySelectorAll('iframe')
        for iframe in iframes:
            content = await iframe.contentFrame()
            expandable_contents.append({
                'type': 'iframe',
                'content': content
            })
        await page.browser().close()
        return expandable_contents

    def retrieve_text(self, page_source):
        soup = BeautifulSoup(page_source, 'html.parser')
        title = ''
        body = ''

        doc = Document(page_source)
        title = doc.title()

        for element in soup.find_all(['p', 'div', 'span']):
            if 'nav' in element.get('class', []) or 'footer' in element.get('class', []) or element.find_parents(['nav', 'footer']):
                continue
            body += element.get_text(separator=' ', strip=True) + '\n'

        return title, body.strip()

    def digest_data(self, data):
        # Placeholder implementation for digest_data
        pass
