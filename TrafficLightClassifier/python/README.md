# Traffic-Light-Classifier

## Requirements.
Achieve 90% accuracy.  
Never Classify Red as Green or Green as Red.  
Use OpenCV.  
## Results.
I achieved a 95% accuracy.  
And passed the project.
## How does it work?
Images are first standardised to 32*32 and labelled.  
Both Features crop the image in half -> the Green feature crops the top half and the red feature crops the bottom half.  
The cropped image is then converted into the HSV colorspace from RGB.  
The average saturation of the HSV image is calculated -> this allows for adjusting the saturation value of the lower mask for each image.  
Cropped images are then masked and put through a gaussian blur.
And finally the average brightness of the masked image is returned.  
A comparison between the values of the red and green feature of an image estimate the color of the traffic light.