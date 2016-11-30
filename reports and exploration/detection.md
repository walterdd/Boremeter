# Анализ и исследование методов и библиотек для детектинга лиц


## Сравнение AUC-ROC разных детектеров:
<img src="http://www.cbsr.ia.ac.cn/faceevaluation/images/figures/curves/whole.png" width="600">

## [OpenCV haar cascades](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)

<img src="http://docs.opencv.org/trunk/haar_features.jpg" width="224">
<img src="http://docs.opencv.org/trunk/haar.png" width="224">

Плюсы:
+ питон
+ быстро (очень)
+ большой рекол
+ можно легко менять пороги каскада классификаторов

Минусы
- много шумов
- подбор параметров сильно влияет на скорость и качество
- работает сильно медленнее при малом maxScale


## [Dlib](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) (HOG Cascade detector)

HOG Cascade detector:

<img src="http://1.bp.blogspot.com/-pPgDErLVJ_k/UvBGZk22ZXI/AAAAAAAAALs/c0mJmAVZnQE/s1600/face_fhog_filters.png" width="224">

Плюсы:
+ апи для питона
+ [точность намного лучше, чем у openCV](https://www.youtube.com/watch?v=LsK0hzcEyHI)

Минусы
- медленно
- маленький рекол


__________________

Другие ссылки:

https://github.com/menpo/menpodetect/ - питон, поверх длиб и опенсв

http://jiansun.org/papers/ECCV14_JointCascade.pdf - детектор

http://markusmathias.bitbucket.org/2014_eccv_face_detection/ - плюсы, матлаб хорошо детектит 

https://github.com/ShiqiYu/libfacedetection - библиотека непонятно на чем, но много лойсов

https://github.com/TadasBaltrusaitis/OpenFace - плюсы, детект и анализ

https://github.com/RiweiChen/DeepFace - дичайше крутая, но долгая штука для анализа (caffe, python)

https://bitbucket.org/rodrigob/doppia - плюсы, говорят, хорошо детектят




