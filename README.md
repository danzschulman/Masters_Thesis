# Dan Schulman's Master's Thesis (Aug 21 2018)
#### Recognizing and Generating Natural Language Referring Expressions in Images
#### Ben Gurion Univeristy of the Negev - The Faculty of Natural Sciences - Department of Computer Science

In this work, we are dealing with the tasks of recognizing and generating natural language referring expression in images.
* In the first task, given an image and a referring expression such as "the zebra on the left", we would like to recognize the actual object ("left zebra") in the image by surrounding it with a bounding-box.
* In the second task, given an image and a bounding-box surrounding one of the objects in it, we would like to generate an unambiguous referring expression describing the object.

You can find more information about my Master's Thesis [here](https://danzschulman.github.io).

The source code for data augmentation can be found [here](https://github.com/danzschulman/refer).

The source code for the improvement of "A Joint Speaker-Listener-Reinforcer Model for Referring Expressions" can be found [here](https://github.com/danzschulman/speaker_listener_reinforcer/tree/resnet).

## Comprehension / Data Augmentation

![alt text](https://github.com/danzschulman/Masters_Thesis/raw/master/data_augmentation_example.png "Data Augmentation Example")

![alt text](https://github.com/danzschulman/Masters_Thesis/raw/master/data_augmentation_results.png "Data Augmentation Results")

## Generation

![alt text](https://github.com/danzschulman/Masters_Thesis/raw/master/generation_results.png "Data Generation Results")
