# MaskNet
## 2020F-IntroductionToDeepLearning-Term Project

### Google Docs Report
https://docs.google.com/document/d/1B5TG_wnZ17lB8bDstGU-ddWnLBjZRIcZZVb1spP59Co/edit

### YOLO 모델은 다른 repository에 저장해 두었습니다.

---
### Requirement & Environment Setting

Google Colab에서 training한 weight를 사용하였기 때문에 Model Loading 함수가 포함된 코드(ex) Caffe_DNN_Video.py)를 실행하실 때에는 tensorflow-gpu==2.4.0 이 설치되어 있어야 합니다.

다만, MTCNN 라이브러리의 경우에는 tf 2.4에서 작동하지 않는 오류가 있기 때문에 이 경우에는 tf 2.2로 downgrade 하셔서 코드를 실행하시길 바랍니다.

MTCNN 라이브러리가 포함된 코드

1. MTCNN_face_detector

환경 전환의 용이함을 위해서 anaconda나 pyenv를 이용하셔서 두 개의 가상환경을 만들어서 실행하시길 추천 드립니다.

requirements.txt file은 tensorflow-gpu 2.4.0 기준으로 작성되었습니다.

---
### Execution Two-Step Detector

현재 디렉토리에서 python Caffe_DNN_Video.py를 실행하시면 되고, 웹캠이 설치된 환경에서만 작동을 합니다.

또한, 실행 테스트를 GPU(CUDA 11.0)가 설치된 환경에서 하였기 때문에, 다른 환경에서의 테스트는 하지 못한 상황입니다.

---
### Reference

1. Face Mask Detection Dataset, https://www.kaggle.com/omkargurav/face-mask-dataset
    
1. MaskedFace-Net, https://github.com/cabani/MaskedFace-Net

1. YOLO dataset, https://github.com/iAmEthanMai/mask-detection-dataset.git
    
1. Mobilnet Fine Tuning Reference Code, https://github.com/gachonyws/face-mask-detector
    
1. HaarCascade Test Code, https://jinho-study.tistory.com/230
    
1. HaarCascade Pre-trained weight, https://github.com/opencv/opencv/tree/master/data/haarcascades
    
1. MTCNN Test Code, https://github.com/ipazc/mtcnn
    
1. Caffe DNN Test Code, https://github.com/gachonyws/face-mask-detector
    
1. Caffe DNN Pre-trained weight, https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel
    
1. YOLO Fine Tuning, https://medium.com/analytics-vidhya/covid-19-face-mask-detection-using-yolov5-8687e5942c81
    
1. Multi-task Cascaded Convolutional Networks (MTCNN) for Face Detection and Facial Landmark Alignment, https://medium.com/@iselagradilla94/multi-task-cascaded-convolutional-networks-mtcnn-for-face-detection-and-facial-landmark-alignment-7c21e8007923

---

1. caffe_dnn_module

    Caffe DNN Face Detction 모델의 weight를 저장해 둔 곳입니다.
    
    Reference : https://github.com/gopinath-balu/computer_vision

1. checkpoint

    Train 된 model들을 저장해 둔 폴더입니다.
    
    Parameter 수가 많았던 CNN 모델은 github에 push가 불가한 관계로 etl에 업로드한 zip file에서 확인하실 수 있습니다.

1. colab_codes

    Google Colaboratory에서 학습을 위해 만든 파일들입니다. 파일을 열어보시면 training log가 남아 있어서 이를 확인하실 수 있습니다.

1. data

    학습에 사용된 dataset입니다. 

    data1에 들어있는 데이터를 이용하여 CNN을 학습시켰고, data2 같은 경우에는 이미지에 포함된 마스크의 종류가 모두 덴탈마스크 종류라서 일반적인 경우에는 잘 작동하지 않아 배제하였습니다.

    testdata에 들어있는 데이터들은 저희가 실제로 촬영하여 모델이 잘 작동하는 지 확인하기 위한 데이터들입니다.
    
1. model_summary
    
    model.summary() 코드를 사용해 나온 결과를 txt 파일로 저장한 결과물들입니다.

1. training_log

    Learning curve plot들을 저장해 둔 폴더입니다.

1. Codes

    1. CNN.ipynb
        
        CNN model을 Keras Sequential을 이용해서 Scratch 부터 짠 코드입니다.
    
    1. Caffe_DNN.ipynb
        
        Caffe DNN Face Detection 모델을 테스트 하기 위해 사용한 코드입니다.
    
    1. Caffe_DNN_Video.ipynb
        
        Caffe DNN Face Detection 모델과 저희가 구현한 Lightweight CNN 모델을 이용해 구현한 Two-step Detection Model 코드입니다.
        
    1. HaarCascade_face_detection.py
    
        OpenCV에서 기본으로 제공하는 얼굴 인식 모델인 HaarCascade 모델을 테스트 하기 위해 사용한 코드입니다.

    1. LightweightCNN.ipynb
    
        저희가 구현한 LightweightCNN 모델 코드입니다.
        
    1. MTCNN_face_detection.ipynb
    
        MTCNN 라이브러리에서 제공하는 MTCNN 모델을 테스트 하기 위해 사용한 코드입니다.
        
    1. Mobilenet.ipynb
    
        Keras 안에 구현되어 있는 MobilenetV2를 이용하여 Fine Tuning 하는 코드입니다.
    
    1. utils.py
      
      Data loader, plotting 함수가 구현되어 있습니다.
