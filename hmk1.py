# special IPython command to prepare the notebook for matplotlib

from fnmatch import fnmatch

import numpy as np
import pandas as pd
import matplotlib
import pylab as plt
import requests
from pattern import web


# set some nicer defaults for matplotlib
from matplotlib import rcParams


#these colors come from colorbrewer2.org. Each is an RGB triplet
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.4, 0.4, 0.4)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = True
rcParams['axes.facecolor'] = '#eeeeee'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'


def get_poll_xml(poll_id):
    url = "http://charts.realclearpolitics.com/charts/" + str(poll_id) + ".xml"
    
    result = requests.get(url).text.encode("utf8", "ignore")
    return result

def rcp_poll_data(xml):
    import re
    date_text = re.findall("<series>(.*)<graph gid='1'", xml)[0]
    approve_text = re.findall("<graph gid='1'(.*)<graph gid='2'", xml)[0]
    disapprove_text = re.findall("<graph gid='2'(.*)", xml)[0]

    dates = re.findall("<value xid=.*?>(.*?)</value>", date_text)
    approve = re.findall("<value xid=.*?>(.*?)</value>", approve_text)
    disapprove = re.findall("<value xid=.*?>(.*?)</value>", disapprove_text)
    titles = ['date'] + re.findall("title='(.*?)'", xml)
    result = pd.DataFrame({titles[0]: pd.to_datetime(dates), titles[1]:
        approve, titles[2]: disapprove})
    result[titles[1]][result[titles[1]] == ''] = None
    result[titles[2]][result[titles[2]] == ''] = None
    result[[titles[1], titles[2]]] = result[[titles[1], titles[2]]].astype(float)
    return result

import re

def _strip(s):
    """This function removes non-letter characters from a word
    
    for example _strip('Hi there!') == 'Hi there'
    """
    return re.sub(r'[\W_]+', '', s)

def plot_colors(xml):
    """
    Given an XML document like the link above, returns a python dictionary
    that maps a graph title to a graph color.
    
    Both the title and color are parsed from attributes of the <graph> tag:
    <graph title="the title", color="#ff0000"> -> {'the title': '#ff0000'}
    
    These colors are in "hex string" format. This page explains them:
    http://coding.smashingmagazine.com/2012/10/04/the-code-side-of-color/
    
    Example
    -------
    >>> plot_colors(get_poll_xml(1044))
    {u'Approve': u'#000000', u'Disapprove': u'#FF0000'}
    """
    dom = web.Element(xml)
    result = {}
    for graph in dom.by_tag('graph'):
        title = _strip(graph.attributes['title'])
        result[title] = graph.attributes['color']
    return result

def poll_plot(poll_id):
    """
    Make a plot of an RCP Poll over time
    
    Parameters
    ----------
    poll_id : int
        An RCP poll identifier
    """

    # hey, you wrote two of these functions. Thanks for that!
    xml = get_poll_xml(poll_id)
    data = rcp_poll_data(xml)
    colors = plot_colors(xml)

    #remove characters like apostrophes
    data = data.rename(columns = {c: _strip(c) for c in data.columns})

    #normalize poll numbers so they add to 100%    
    norm = data[colors.keys()].sum(axis=1) / 100    
    for c in colors.keys():
        data[c] /= norm
    
    for label, color in colors.items():
        plt.plot(data.date, data[label], color=color, label=label)        
        
    plt.xticks(rotation=70)
    plt.legend(loc='best')
    plt.xlabel("Date")
    plt.ylabel("Normalized Poll Percentage")

def find_governor_races(html):
    text = requests.get(html).text.encode("utf8", "ignore")
    urls = re.findall("(/epolls/2014/governor/.*?/.*?.html)",
            text)
    urls = map(lambda x:
    "http://www.realclearpolitics.com"+ x, urls)
    return urls


def race_result(url):
    text = requests.get(url).text.encode("utf8", "ignore")
    text = '''<th>Sample</th><th>MoE</th><th>Cutler (I)</th><th>LePage (R)</th><th>Michaud (D)</th><th class="spread">Spread</th></tr><tr class="final"><td class="noCenter">Final Results</td><td>--</td><td>--</td><td>--</td><td>8.4</td><td>48.2</td><td>43.3</td>''' 
    result = re.findall('''Sample.*<th>(.*?) \([RD]\)</th><th>(.*?) \([RD]\)</th>.*?Spread.*?Final Results.*?<td>(\d{2}\.\d{0,1})</td><td>(\d{2}\.\d{0,1})</td>''', text)[0]
    return {result[0]: result[2], result[1]: result[3]}

def id_from_url(url):
    """Given a URL, look up the RCP identifier number"""
    return url.split('-')[-1].split('.html')[0]


def plot_race(url):
    """Make a plot summarizing a senate race
    
    Overplots the actual race results as dashed horizontal lines
    """
    #hey, thanks again for these functions!
    id = id_from_url(url)
    xml = get_poll_xml(id)    
    colors = plot_colors(xml)

    if len(colors) == 0:
        return
    
    #really, you shouldn't have
    result = race_result(url)
    
    poll_plot(id)
    plt.xlabel("Date")
    plt.ylabel("Polling Percentage")
    for r in result:
        plt.axhline(result[r], color=colors[_strip(r)], alpha=0.6, ls='--')

'''
page = 'http://www.realclearpolitics.com/epolls/2014/governor/2014_elections_governor_map.html'

for race in find_governor_races(page):
    plot_race(race)
    plt.show()
'''


