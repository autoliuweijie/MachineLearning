'''
    this is an example of using k-mean to find place
    @author: Liu Weijie
'''
import urllib


if __name__ == '__main__':
    api_system = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'