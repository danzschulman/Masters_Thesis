# Dan Schulman's Master's Thesis (Aug 21 2018)
#### Recognizing and Generating Natural Language Referring Expressions in Images
#### Ben Gurion Univeristy of the Negev - The Faculty of Natural Sciences - Department of Computer Science

In this work, we are dealing with the tasks of recognizing and generating natural language referring expression in images.
* In the first task, given an image and a referring expression such as "the zebra on the left", we would like to recognize the actual object ("left zebra") in the image by surrounding it with a bounding-box.
* In the second task, given an image and a bounding-box surrounding one of the objects in it, we would like to generate an unambiguous referring expression describing the object.

You can find more information about my Master's Thesis [here](https://danzschulman.github.io).

The source code for data augmentation can be found [here](https://github.com/danzschulman/refer).

The source code for the improvement of "A Joint Speaker-Listener-Reinforcer Model for Referring Expressions" can be found [here](https://github.com/danzschulman/speaker_listener_reinforcer/tree/resnet).

Our baselines (decribed below) uses [ReferIt](https://github.com/danzschulman/refer) as the dataset and also [MaskRCNN](https://github.com/matterport/Mask_RCNN) as an object detector.

## Comprehension / Data Augmentation

In this task, we improved the performance of the current state-of-the-art network named MAttNet, by simply flipping all images left-to-right and updateing the referring expressions accordingly. This results is a new dataset, still valid and twice as big.

![alt text](https://github.com/danzschulman/Masters_Thesis/raw/master/data_augmentation_example.png "Data Augmentation Example")

![alt text](https://github.com/danzschulman/Masters_Thesis/raw/master/data_augmentation_results.png "Data Augmentation Results")

## Generation

In this task, we improved the "A Joint Speaker-Listener-Reinforcer Model for Referring Expressions" (SLR) network by replacing the visual encoder (VGG) with a better one (ResNet).

As a baseline, we ran MaskRCNN, detected the bounding-box with IoU >= 0.5 (Intersection-over-Union), then sorted all objects of this type by their x-center. Finally, according to the correct bounding-box location in the sorted list of objects, we created a referring expression such as "leftmost zebra", "second zebra from the right/left" and so on.

![alt text](https://github.com/danzschulman/Masters_Thesis/raw/master/generation_results.png "Data Generation Results")
