Task 3 - 2D computer vision - Darts game analysis

# Algorithm
The main idea of the algorithm is to find the center of the dartboard and the tips of darts pieces. Then after calculating the respective distances we can determine ho many points each dart piece scores

For for this we need to 
1. Crop the board and find its center
2. Find positions of the dart tips

## Crop the board
Algorith is straitforward for this task:
1. Find the bounding box of the biggest contour on an image by area
2. Crop the bounding box and resize image for convenience

Now we can easily find the center of a board


## Dart Tips
To find dart tips the following algorithm has been created:
1. Find the darts themselves. It can be done easily because their color is well distinguished from the board
2. Sometimes 2 darts of one color overlap and share same contour. So before finding the actual contours, we need to erode our masks
3. After erosion, calculate contours of individual dart pieces
![alt text](![alt text](https://github.com/dnsshplk/Task3_int/blob/3100a1dacea783c7eaf5a5f84e921acd30b46d72/algo_illustrations/contours_ellipses.png))

4. We need to find the tips of darts, but so far we have only contours of their tails. To find tip, fit an ellipse to a contour, get ellipse center and adjust ut based on the elipse orientation and contours area. 
    - If an area of a contour is small, we are looking at the dart right from behind and we do not need to adjust the center. 
    - If an area of a contour is medium, we found 2 of 4 tailwings of a dart and need to move along the smaller radius upward
    - If an area of a contour is large, we found nearly whole dart piece and need to move along the bigger radius upward

## Calculate scores
When we have the center of a board and the tips of datrs pieces, we can easily calculate the scores of each team and determine the winned.

# Results Analysis

## Plain Results
The algorithm correcly identified scores of 17 out of 18 pieces. And correctly indentified the winned in all 3 games.
![alt text](https://github.com/dnsshplk/Task3_int/blob/47e9b04c37f4161e4d6257f4176bd820438ec736/results/result_IMG_20240510_172748.jpg)
![alt text](https://github.com/dnsshplk/Task3_int/blob/47e9b04c37f4161e4d6257f4176bd820438ec736/results/result_IMG_20240510_172837.jpg)
![alt text](https://github.com/dnsshplk/Task3_int/blob/47e9b04c37f4161e4d6257f4176bd820438ec736/results/result_IMG_20240510_172930.jpg)


## On Distortion
The only place, where the algotithm takes into account distortion of an image, is when a little offset +10 is added to the center of cropped board. Experiments showed that this is more robust than projecting ellipse (the actual shape of a board) into the cirle and then cropping it (it is much more expensive and does not impove results). + the distortion itself is pretty small (<2% difference on y-axis and even less on x-axis)

## Calculation darts tips positions
The formula(for adjustig a center of an ellipse of some dart contour) affects results much more that distortion. The formula was created empirically. Of course, we can find such constants that all scores are correst, but I guess it would be somewhat like manual "overfitting" the data.
