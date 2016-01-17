from collections import defaultdict
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as plb

from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib as mpl

#colorbrewer2 Dark2 qualitative color table
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()
        
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

#this mapping between states and abbreviations will come in handy later
states_abbrev = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

#load in state geometry
state2poly = defaultdict(list)

data = json.load(file("data/us-states.json"))
for f in data['features']:
    state = states_abbrev[f['id']]
    geo = f['geometry']
    if geo['type'] == 'Polygon':
        for coords in geo['coordinates']:
            state2poly[state].append(coords)
    elif geo['type'] == 'MultiPolygon':
        for polygon in geo['coordinates']:
            state2poly[state].extend(polygon)

            
def draw_state(plot, stateid, **kwargs):
    """
    draw_state(plot, stateid, color=..., **kwargs)
    
    Automatically draws a filled shape representing the state in
    subplot.
    The color keyword argument specifies the fill color.  It accepts keyword
    arguments that plot() accepts
    """
    for polygon in state2poly[stateid]:
        xs, ys = zip(*polygon)
        plot.fill(xs, ys, **kwargs)

        
def make_map(states, label):
    """
    Draw a cloropleth map, that maps data onto the United States
    
    Inputs
    -------
    states : Column of a DataFrame
        The value for each state, to display on a map
    label : str
        Label of the color bar

    Returns
    --------
    The map
    """
    fig = plt.figure(figsize=(12, 9))
    ax = plt.gca()

    if states.max() < 2: # colormap for election probabilities 
        cmap = cm.RdBu
        vmin, vmax = 0, 1
    else:  # colormap for electoral votes
        cmap = cm.binary
        vmin, vmax = 0, states.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    skip = set(['National', 'District of Columbia', 'Guam', 'Puerto Rico',
                'Virgin Islands', 'American Samoa', 'Northern Mariana Islands'])
    for state in states_abbrev.values():
        if state in skip:
            continue
        color = cmap(norm(states.ix[state]))
        draw_state(ax, state, color = color, ec='k')

    #add an inset colorbar
    ax1 = fig.add_axes([0.45, 0.70, 0.4, 0.02])    
    cb1=mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                  norm=norm,
                                  orientation='horizontal')
    ax1.set_title(label)
    remove_border(ax, left=False, bottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-180, -60)
    ax.set_ylim(15, 75)
    a = raw_input()
    return ax

import datetime
today = datetime.datetime(2012, 10, 2)

electoral_votes = pd.read_csv("data/electoral_votes.csv").set_index('State')

#make_map(electoral_votes.Votes, "Electoral Votes")
predictwise = pd.read_csv('data/predictwise.csv').set_index('States')
#make_map(predictwise['Obama'], "Obama Win Probability")

import random

def simulate_election(model, nsim):
    def simulation(model):
        model['simulation'] = (random.random() < model['Obama']) * model['Votes']
        return model['simulation'].sum()
    return np.array([simulation(model) for _ in range(nsim)])

#print simulate_election(predictwise, 1000)

def plot_simulation(simulation):
    fig, axes = plb.subplots()
    axes.set(xlim=(240, 380), ylim=(0, .2), xlabel='Obama Electoral College Votes', ylabel='Probability', title='Chance of Obama Victory')
    plb.hist(simulation,bins=np.arange(240, 380,
        2), normed=True, label="Simulations")
    plb.axvline(330, ymax=.5, color="red", label="Actual Outcome")
    plb.axvline(269, ymax=.5, color="black", label="Victory Threshold")
    plb.show(block=True)

#plot_simulation(simulate_election(predictwise, 10000))

gallup_2012=pd.read_csv("data/g12.csv").set_index('State')
gallup_2012["Unknown"] = 100 - gallup_2012.Democrat - gallup_2012.Republican

def simple_gallup_model(gallup):
    gallup['Obama'] = (gallup['Dem_Adv'] > 0) * 1
    return gallup

def uncertain_gallup_model(gallup):
    import scipy.stats
    gallup['Obama'] = scipy.stats.norm(gallup['Dem_Adv'], 3).cdf(0)
    print gallup.head()
    return gallup

'''
model = uncertain_gallup_model(gallup_2012)
model = model.join(electoral_votes)
prediction = simulate_election(model, 1000)
plot_simulation(prediction)
make_map(model.Obama, "P(Obama): Simple Model")
'''

def biased_gallup_model(gallup, bias):
    import scipy.stats
    gallup['Obama'] = scipy.stats.norm(-1 * gallup['Dem_Adv'] + bias, 3).cdf(0)
    print gallup.head()
    return gallup

gallup_08 = pd.read_csv("data/g08.csv").set_index('State')
results_08 = pd.read_csv('data/2008results.csv').set_index('State')

prediction_08 = gallup_08[['Dem_Adv']]
prediction_08['Dem_Win']=results_08["Obama Pct"] - results_08["McCain Pct"]

'''
import numpy.polynomial.polynomial as poly
fig, axes = plb.subplots()
axes.set(xlabel = 'Dem Advantage', ylabel = 'Dem Win', xlim = (-40,40), ylim = (-40, 40))
plb.scatter(prediction_08['Dem_Adv'], prediction_08['Dem_Win']) 
fit = poly.polyfit(prediction_08['Dem_Adv'], prediction_08['Dem_Win'], 1)
y1, y2 = fit[0] + -40*fit[1], fit[0] + 40*fit[1]
plb.plot([-40, 40], [y1, y2])
plb.show(block=True)

#print prediction_08[(prediction_08['Dem_Win'] < 0) & (prediction_08['Dem_Adv']
#    > 0)].index.values
bias = (prediction_08.Dem_Adv - prediction_08.Dem_Win).mean()

model = biased_gallup_model(gallup_2012, bias)
model = model.join(electoral_votes)
prediction = simulate_election(model, 1000)
plot_simulation(prediction)
make_map(model.Obama, "P(Obama): Simple Model")
'''

national_results=pd.read_csv("data/nat.csv")
national_results.set_index('Year',inplace=True)

polls04=pd.read_csv("data/p04.csv")
polls04.State=polls04.State.replace(states_abbrev)
polls04.set_index("State", inplace=True);

pvi08=polls04.Dem - polls04.Rep - (national_results.xs(2004)['Dem'] -
        national_results.xs(2004)['Rep'])

e2008 = pd.DataFrame(pvi08)
e2008.columns = ['pvi']
e2008['Dem_Adv'] = prediction_08['Dem_Win'] - prediction_08['Dem_Win'].mean()
e2008['obama_win'] = (prediction_08['Dem_Win'] > 0) * 1
e2008['Dem_Win'] = prediction_08['Dem_Win']

pvi12 = e2008.Dem_Win - (national_results.xs(2008)['Dem'] -
        national_results.xs(2008)['Rep'])
e2012 = pd.DataFrame(dict(pvi=pvi12, Dem_Adv=gallup_2012.Dem_Adv -
    gallup_2012.Dem_Adv.mean()))
e2012 = e2012.sort_index()

results2012 = pd.read_csv("data/2012results.csv")
results2012.set_index("State", inplace=True)
results2012 = results2012.sort_index()

'''
ax = plb.subplot()
plb.scatter(e2008['pvi'], e2008['Dem_Adv'], color = map(lambda x: 'b' if x
    else 'r', e2008['obama_win']))
plb.scatter(e2012['pvi'], e2012['Dem_Adv'], color='grey')
ax.set(xlabel="pvi08", ylabel="Dem_Adv")
plb.show(block=True)
'''

from sklearn.linear_model import LogisticRegression

def prepare_features(frame2008, featureslist):
    y= frame2008.obama_win.values
    X = frame2008[featureslist].values
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return y, X

def fit_logistic(frame2008, frame2012, featureslist, reg=0.0001):
    y, X = prepare_features(frame2008, featureslist)
    clf2 = LogisticRegression(C=reg)
    clf2.fit(X, y)
    X_new = frame2012[featureslist]
    obama_probs = clf2.predict_proba(X_new)[:, 1]
    
    df = pd.DataFrame(index=frame2012.index)
    df['Obama'] = obama_probs
    return df, clf2

from sklearn.grid_search import GridSearchCV

def cv_optimize(frame2008, featureslist, n_folds=10, num_p=100):
    y, X = prepare_features(frame2008, featureslist)
    clf = LogisticRegression()
    parameters = {"C": np.logspace(-4, 3, num=num_p)}
    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds)
    gs.fit(X, y)
    return gs.best_params_, gs.best_score_

def cv_and_fit(frame2008, frame2012, featureslist, n_folds=5):
    bp, bs = cv_optimize(frame2008, featureslist, n_folds=n_folds)
    predict, clf = fit_logistic(frame2008, frame2012, featureslist, reg=bp['C'])
    return predict, clf

df, clf = cv_and_fit(e2008, e2012, ['Dem_Adv', 'pvi'], n_folds=5)

'''
model = df
model = model.join(electoral_votes)
prediction = simulate_election(model, 1000)
plot_simulation(prediction)
make_map(model.Obama, "P(Obama): Simple Model")
'''

from matplotlib.colors import ListedColormap
def points_plot(e2008, e2012, clf):
    """
    e2008: The e2008 data
    e2012: The e2012 data
    clf: classifier
    """
    Xtrain = e2008[['Dem_Adv', 'pvi']].values
    Xtest = e2012[['Dem_Adv', 'pvi']].values
    ytrain = e2008['obama_win'].values == 1
    
    X=np.concatenate((Xtrain, Xtest))
    
    # evenly sampled points
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    #plot background colors
    ax = plt.gca()
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    cs = ax.contourf(xx, yy, Z, cmap='RdBu', alpha=.5)
    cs2 = ax.contour(xx, yy, Z, cmap='RdBu', alpha=.5)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize=14)
    
    # Plot the 2008 points
    ax.plot(Xtrain[ytrain == 0, 0], Xtrain[ytrain == 0, 1], 'ro', label='2008 McCain')
    ax.plot(Xtrain[ytrain == 1, 0], Xtrain[ytrain == 1, 1], 'bo', label='2008 Obama')
        
    # and the 2012 points
    ax.scatter(Xtest[:, 0], Xtest[:, 1], c='k', marker="s", s=50, facecolors="k", alpha=.5, label='2012')
    plt.legend(loc='upper left', scatterpoints=1, numpoints=1)

    a = raw_input()
    return ax

#points_plot(e2008, e2012, clf)


multipoll = pd.read_csv('data/cleaned-state_data2012.csv', index_col=0)

#convert state abbreviation to full name
multipoll.State.replace(states_abbrev, inplace=True)

#convert dates from strings to date objects, and compute midpoint
multipoll.start_date = multipoll.start_date.apply(pd.to_datetime)
multipoll.end_date = multipoll.end_date.apply(pd.to_datetime)
multipoll['poll_date'] = multipoll.start_date + (multipoll.end_date - multipoll.start_date).values / 2

#compute the poll age relative to Oct 2, in days
multipoll['age_days'] = (today - multipoll['poll_date']).values / np.timedelta64(1, 'D')

#drop any rows with data from after oct 2
multipoll = multipoll[multipoll.age_days > 0]

#drop unneeded columns
multipoll = multipoll.drop(['Date', 'start_date', 'end_date', 'Spread'], axis=1)

#add electoral vote counts
multipoll = multipoll.join(electoral_votes, on='State')

#drop rows with missing data
multipoll.dropna()

def state_average(multipoll):
    N = multipoll.groupby('State').size()
    poll_mean = multipoll.groupby('State')['obama_spread'].mean()
    poll_std = multipoll.groupby('State')['obama_spread'].std()
    averages = pd.concat([N, poll_mean, poll_std], axis=1)
    averages.columns = ['N', 'poll_mean', 'poll_std']
    return averages

avg = state_average(multipoll).join(electoral_votes, how='outer')


def default_missing(results):
    red_states = ["Alabama", "Alaska", "Arkansas", "Idaho", "Wyoming"]
    blue_states = ["Delaware", "District of Columbia", "Hawaii"]
    results.ix[red_states, ["poll_mean"]] = -100.0
    results.ix[red_states, ["poll_std"]] = 0.1
    results.ix[blue_states, ["poll_mean"]] = 100.0
    results.ix[blue_states, ["poll_std"]] = 0.1
default_missing(avg)

def aggregated_poll_model(polls):
    import scipy.stats
    df = pd.DataFrame(polls['Votes'])
    df['Obama'] = np.array(scipy.stats.norm(polls['poll_mean'],
        polls['poll_std']).cdf(0))
    return df

'''
model = aggregated_poll_model(avg)
prediction = simulate_election(model, 1000)
plot_simulation(prediction)
make_map(model.Obama, "P(Obama): Simple Model")
'''


def weighted_state_average(df):
    N = df.groupby('State').size()
    poll_mean = df.groupby('State')['obama_spread'].mean()
    poll_std = df.groupby('State')['obama_spread'].std()
    result = pd.concat([N, poll_mean, poll_std], axis=1)
    result.columns = ['N', 'poll_mean', 'poll_std']
    return result

new_avg = weighted_state_average(multipoll)
default_missing(new_avg)

model = aggregated_poll_model(new_avg)
prediction = simulate_election(model, 1000)
plot_simulation(prediction)
make_map(model.Obama, "P(Obama): Simple Model")
