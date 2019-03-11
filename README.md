# Pokemon

## Introduction

Which Pokemon is going to win ?

In this project we are going to create a neural network with Tensorflow to
be able to predict the winner of a dual between two Pokemon.

## Data

Data are coming from a Kaggle dataset.
You can find it [here](https://www.kaggle.com/terminus7/pokemon-challenge#tests.csv)

The begining is a transformation of *pokemon.csv* by changing each *NaN* 
value by *NoT*, value which is going to be treat latter.

There is a first script call *all.py*, which is in *lib* directory, which
merge *combat.csv* and *pokemonC.csv* too create a big CSV file called *full.csv*.

## Train

In *lib* repository, you will see many *train* file with numbers. This numbers
are a describtion of the neural network code in. For exemple, *train4_1.py*
contain a neural network where the input layer is compose by 4 neurones
and an output layer with one neurones (no hidder layer).

To run a train you must specify the location of *full.csv* file :
./train10_2.py ../data/full.csv

## Visualisation

Please run tensorboard in direcorty */tmp/tensorflow/graph/*
exemple :
python ~/.local/lib/python2.7/site-packages/tensorboard/main.py --logdir /tmp/tensorflow/graph/
