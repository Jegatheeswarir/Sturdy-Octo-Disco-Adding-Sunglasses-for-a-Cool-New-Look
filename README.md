# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## Program :
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Face Image
faceImage = cv2.imread('cvv.png')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
```
<img width="558" height="729" alt="Screenshot 2025-09-07 231635" src="https://github.com/user-attachments/assets/4a076183-6d5b-4f35-bf99-f75c3ef3fecd" />

```
faceImage.shape
```
<img width="168" height="61" alt="image" src="https://github.com/user-attachments/assets/29963502-e5da-45fb-83d5-9870f3b4e64d" />

```
glassPNG = cv2.imread('glass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
```

<img width="883" height="508" alt="image" src="https://github.com/user-attachments/assets/54f7e089-d1d3-4795-9684-873efa9c849e" />

```
glassPNG = cv2.resize(glassPNG,(190,50))
print("image Dimension ={}".format(glassPNG.shape))
```

<img width="233" height="32" alt="image" src="https://github.com/user-attachments/assets/7616b70b-8a42-4334-9b3a-b1619837e305" />

```
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```

<img width="544" height="105" alt="image" src="https://github.com/user-attachments/assets/0107c86d-630f-4f34-a3c7-af4c6cc5a2a6" />

```
faceWithGlassesNaive = faceImage.copy()
y1, y2 = 180, 255   
x1, x2 = 70, 300   
glassResized = cv2.resize(glassBGR, (x2 - x1, y2 - y1))  
faceWithGlassesNaive[y1:y2, x1:x2] = glassResized
plt.imshow(faceWithGlassesNaive[..., ::-1])
```

<img width="403" height="428" alt="image" src="https://github.com/user-attachments/assets/98b08f39-2b21-4b86-ac7a-f3e0a7d0426f" />

```
y1, y2 = 180, 255   
x1, x2 = 70, 300   

eyeROI = faceImage[y1:y2, x1:x2]

glassResized   = cv2.resize(glassBGR, (x2 - x1, y2 - y1))       
glassMask1Resized = cv2.resize(glassMask1, (x2 - x1, y2 - y1))   

glassMask = cv2.merge([glassMask1Resized]*3).astype(np.float32) / 255.0

eyeROI_f      = eyeROI.astype(np.float32)
glassResized_f = glassResized.astype(np.float32)

maskedEye   = cv2.multiply(eyeROI_f, (1.0 - glassMask))
maskedGlass = cv2.multiply(glassResized_f, glassMask)

eyeRoiFinal = cv2.add(maskedEye, maskedGlass).astype(np.uint8)

faceWithGlassesArithmetic = faceImage.copy()
faceWithGlassesArithmetic[y1:y2, x1:x2] = eyeRoiFinal

plt.figure(figsize=[20,20])
plt.subplot(131); plt.imshow(maskedEye[..., ::-1].astype(np.uint8)); plt.title("Masked Eye Region")
plt.subplot(132); plt.imshow(maskedGlass[..., ::-1].astype(np.uint8)); plt.title("Masked Sunglass Region")
plt.subplot(133); plt.imshow(faceWithGlassesArithmetic[..., ::-1]); plt.title("Augmented Face with Sunglass")
plt.show()
```

<img width="585" height="226" alt="image" src="https://github.com/user-attachments/assets/296e7868-62d4-46d6-857c-152516fb7e21" />

```
faceWithGlassesArithmetic[180:255,70:300] = eyeRoiFinal
plt.figure(figsize=[20,20])
plt.subplot(121); plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image")
plt.subplot(122); plt.imshow(faceWithGlassesArithmetic[:,:,::-1]); plt.title("With Sunglass")
```

<img width="548" height="370" alt="image" src="https://github.com/user-attachments/assets/9706f1ec-fcde-4f1d-a625-2431310115e5" />

Feel free to fork, contribute, or customize this project for your creative needs!
