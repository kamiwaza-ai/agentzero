from abc import ABC, abstractmethod

class WebInterface(ABC):

    @abstractmethod
    def search(self, query):
        pass

    @abstractmethod
    def category_search(self, category):
        pass

    @abstractmethod
    def load_page(self, url):
        pass

    @abstractmethod
    def retrieve_text(self):
        pass

    @abstractmethod
    def check_expandable_content(self):
        pass

    @abstractmethod
    def digest_data(self, data):
        pass