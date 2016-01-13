import pandas as pd
import numpy as np
import random


def simulate_prizedoor(nsim):
    return np.array([random.randint(0, 2) for _ in range(nsim)])


def simulate_guess(nsim):
    return np.array([0] * nsim)


def goat_door(prizedoors, guesses):
    doors = []
    print prizedoors
    for prize, guess in zip(prizedoors, guesses):
        possibilities = [0, 1, 2]
        possibilities.remove(prize)
        if prize != guess:
            possibilities.remove(guess)
        doors.append(random.choice(possibilities))
    return np.array(doors)


def switch_guess(guesses, goatdoors):
    doors = []
    for guess, goatdoor in zip(guesses, goatdoors):
        possibilities = [0, 1, 2]
        possibilities.remove(guess)
        if guess != goatdoor:
            possibilities.remove(goatdoor)
        doors.append(random.choice(possibilities))
    return np.array(doors)


def win_percentage(guesses, prizedoors):
    return np.sum(np.equal(guesses, prizedoors)) * 100/len(guesses)

guesses = simulate_guess(10000)
prizes = simulate_prizedoor(10000)
goatdoors = goat_door(prizes, guesses)
switches = switch_guess(guesses, goatdoors)

print "No switching win percentage: " + str(win_percentage(guesses, prizes))
print "Switching win percentage: " + str(win_percentage(switches, prizes))
