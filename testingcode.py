from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
model=load_model("model.h5")
classes=["Fibrosis","Normal","Pneumonia"]
img_path=input("enter the path of the image to test:")
img=image.load_img(img_path,target_size=(128,128))
img_array=image.img_to_array(img)
img_array=img_array/255.0
img_array=np.expand_dims(img_array,axis=0)
result=model.predict(img_array)
predicted_class=np.argmax(result)
print("prediction:",classes[predicted_class])
print("Confidence:",np.max(result)*100,"%")
h=classes[predicted_class]
font=cv2.FONT_HERSHEY_DUPLEX
img=cv2.imread(img_path)
cv2.putText(img,h,(10,30),font,1.0,(0,0,255),1)
cv2.imshow(h,img)
cv2.waitKey(0)
cv2.imshow()
print(h)