#!/usr/bin/env python2.7
#-*- encoding: utf-8 -*-




import pandas as pd
import numpy as np

def bina(first, second, win):
    if first == win:
        return 0
    else:
        return 1

def change(win):
    if int(win) == 1:
        return 0
    else:
        return 1

def main():
    """
    main function
    """
    pokemon = pd.read_csv('pokemonC.csv')
    combat = pd.read_csv('combats.csv')
    tests = pd.read_csv('tests.csv')

    combat['Winner'] = combat.apply(
        lambda row: bina(first=row['First_pokemon'],
                         second=row['Second_pokemon'],
                         win=row['Winner']), axis=1)

    total = []


    col = [combat.columns[0]]
    col2 = [combat.columns[1]]

    for name in pokemon.columns:
        col.append(name +str(0))
        col2.append(name+str(1))

    col.append(combat.columns[1])
    col2.append(combat.columns[0])

    for name in pokemon.columns:
        col.append(name +str(1))
        col2.append(name +str(0))

    col.append(combat.columns[2])
    col2.append(combat.columns[2])
    col.remove('#0')
    col.remove('#1')
    col2.remove('#0')
    col2.remove('#1')

    count = 0
    for index, row in combat.iterrows():
        buff = [count]
        buff.append(row['First_pokemon'])
        first = pokemon.loc[pokemon['#'] == row['First_pokemon']]
        first = first.drop(['#'], axis=1).values[0]

        second = pokemon.loc[pokemon['#'] == row['Second_pokemon']]
        second = second.drop(['#'], axis=1).values[0]

        for value in first:
            buff.append(value)

        buff.append(row['Second_pokemon'])
        for value in second:
            buff.append(value)

        buff.append(row['Winner'])
        total.append(buff)
        count +=1

    total = np.asarray(total)
    df = pd.DataFrame(total[:,1:], index = total[:,0], columns = col)
    df2 = df[col2]


    df2['Winner'] = df2.apply(
        lambda row: change(row['Winner']), axis=1)
    df2.columns = col

    dataframe = pd.concat([df, df2])

    dataframe.to_csv('full.csv', index=False)

if __name__ =='__main__':
    main()

