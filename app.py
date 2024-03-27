from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)
app.static_folder = 'static'

def count_persons(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    (rects, weights) = hog.detectMultiScale(gray, winStride=(5, 5), padding=(8, 8), scale=1.05)

    
    img_detected = img.copy()  
    for (x, y, w, h) in rects:
        cv2.rectangle(img_detected, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    num_persons = len(rects)
    if num_persons >= 5:
        signal = "Turn on the red light"
    else:
        signal = "Turn on the green light"
    
    return img_detected, num_persons, signal


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        if 'image' in request.files:
            image = request.files['image']
            if image:
                try:
                    
                    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
                    
                    
                    img_detected, num_persons, signal = count_persons(img)
                    
                    
                    output_path = 'static/output_image.png'
                    cv2.imwrite(output_path, img_detected)
                    
                    
                    original_image_path = 'static/input_image.png'
                    cv2.imwrite(original_image_path, img)
                    
                    return render_template('index.html', output_image=output_path, input_image=original_image_path, num_persons=num_persons, signal=signal)
                except Exception as e:
                    
                    return render_template('index.html', error_message=str(e))

    return render_template('index.html', output_image=None, input_image=None, num_persons=None, signal=None)

if __name__ == '__main__':
    app.run(debug=True)
