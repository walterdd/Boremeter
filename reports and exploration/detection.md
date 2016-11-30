# Анализ и исследование методов и библиотек для детектинга лиц




## [OpenCV haar cascades](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)

<img src="http://docs.opencv.org/trunk/haar_features.jpg" width="224">
<img src="http://docs.opencv.org/trunk/haar.png" width="224">

Плюсы:
+ быстро (очень)
+ большой рекол

Минусы
- много шумов
- подбор параметров
- работает сильно медленнее при малом maxScale


## [Dlib](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) (HOG Cascade detector)

HOG Cascade detector
<img src="http://1.bp.blogspot.com/-pPgDErLVJ_k/UvBGZk22ZXI/AAAAAAAAALs/c0mJmAVZnQE/s1600/face_fhog_filters.png" width="224">

Плюсы:
+ [точность намного лучше, чем у openCV](https://www.youtube.com/watch?v=LsK0hzcEyHI)

Минусы
- медленно
- маленький рекол








______________________________
Ссылки на почитать: 

[http://wiki.ros.org/face_detection_tracking](http://wiki.ros.org/face_detection_tracking)
