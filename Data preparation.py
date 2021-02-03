import skimage.data
print(skimage.__version__)

#이미지 작업 루트 가져오기
import cv2
import numpy as np
#이미지 처리를 위한 모듈 import 및 전 작업

## 인풋 data와 save data의 path를 지정한다.(class별로 작업을 해야함,os를 이용하여도 무방)
##define Input data path:load path, Save path

##      (first You rename gliches name same object in windows:like whistle, 1400blips)..
##      (as you can see data names changed object(1),object(2),object(3))


for i in range(0,7289):  #7289is the most numerous number of glitches(blips has 7289 glitches)
    #even if there are some error you can see it works properly
    #(error happened when some class have little glitches than 7289)

    #you must do it same process all class and rewrite your load%save path..!!

    load_path01 = 'D:\\GSorigin\\Just_resized1\\Whistle\\Whistle (' + str(4*i+1) +').png'
    load_path02 = 'D:\\GSorigin\\Just_resized2\\Whistle\\Whistle (' + str(4*i+2) + ').png'
    load_path03 = 'D:\\GSorigin\\Just_resized3\\Whistle\\Whistle (' + str(4*i+3) + ').png'
    load_path04 = 'D:\\GSorigin\\Just_resized4\\Whistle\\Whistle (' + str(4*i+4) + ').png'

    save_path01 = 'D:\\GSorigin\\Just_HSV1\\Whistle\\Whistle(' + str(4*i+1) + ').png'
    save_path02 = 'D:\\GSorigin\\Just_HSV2\\Whistle\\Whistle(' + str(4*i+2) + ').png'
    save_path03 = 'D:\\GSorigin\\Just_HSV3\\Whistle\\Whistle(' + str(4*i+3) + ').png'
    save_path04 = 'D:\\GSorigin\\Just_HSV4\\Whistle\\Whistle(' + str(4*i+4) + ').png'
    save_path05 = 'D:\\GSorigin\\Just_HSV_M\\Whistle\\Whistle (' + str(i) + ').png'

    #image read

    image01=cv2.imread(load_path01)
    image02=cv2.imread(load_path02)
    image03=cv2.imread(load_path03)
    image04=cv2.imread(load_path04)

    #image trimming

    image_cut01 = image01[60:570, 85:690]
    image_cut02 = image02[60:570, 85:690]
    image_cut03 = image03[60:570, 85:690]
    image_cut04 = image04[60:570, 85:690]

    #trimmed data fix size,interpolation, and save as image_resized
    image_resized01=cv2.resize(image01,dsize=(47,57),interpolation=cv2.INTER_AREA)
    image_resized02=cv2.resize(image02,dsize=(47,57),interpolation=cv2.INTER_AREA)
    image_resized03=cv2.resize(image03,dsize=(47,57),interpolation=cv2.INTER_AREA)
    image_resized04=cv2.resize(image04,dsize=(47,57),interpolation=cv2.INTER_AREA)

    #(1)HSV channel process(you can skip this process, you can also change boundary number)
    #[H,S,V] boundary 영역 값 지정
    #[H,S,V] Define boundary range

#    lower_blue = np.array([50,100,50]) # HSV
#    upper_blue = np.array([130,255,255])
#    lower_blue = np.array([30,50,50]) # HSV
    lower_blue = np.array([30,50,50]) # HSV
    upper_blue = np.array([180,255,255])

    #(2)HSV channel process(you can skip this process)
    #data read and color channel transform

    image_hsv01 = cv2.cvtColor(image_resized01, cv2.COLOR_RGB2HSV)  # BGR에서 HSV로 변환
    image_hsv02 = cv2.cvtColor(image_resized02, cv2.COLOR_RGB2HSV)  # BGR에서 HSV로 변환
    image_hsv03 = cv2.cvtColor(image_resized03, cv2.COLOR_RGB2HSV)  # BGR에서 HSV로 변환
    image_hsv04 = cv2.cvtColor(image_resized04, cv2.COLOR_RGB2HSV)  # BGR에서 HSV로 변환

    #(3)HSV channel process(you can skip this process)


    mask01 = cv2.inRange(image_hsv01, lower_blue, upper_blue)  # 마스크를 만듭니다.
    image_bgr_masked01 = cv2.bitwise_and(image_resized01, image_resized01, mask=mask01)  # 이미지에 마스크를 적용

    mask02 = cv2.inRange(image_hsv02, lower_blue, upper_blue)  # 마스크를 만듭니다.
    image_bgr_masked02 = cv2.bitwise_and(image_resized02, image_resized02, mask=mask02)  # 이미지에 마스크를 적용

    mask03 = cv2.inRange(image_hsv03, lower_blue, upper_blue)  # 마스크를 만듭니다.
    image_bgr_masked03 = cv2.bitwise_and(image_resized03, image_resized03, mask=mask03)  # 이미지에 마스크를 적용

    mask04 = cv2.inRange(image_hsv04, lower_blue, upper_blue)  # 마스크를 만듭니다.
    image_bgr_masked04 = cv2.bitwise_and(image_resized04, image_resized04, mask=mask04)  # 이미지에 마스크를 적용

    #Save Process
    #If you save without HSV channel Process just change image_brg_maskedXX to image_resizedXX

    cv2.imwrite(save_path01, image_bgr_masked01)
    cv2.imwrite(save_path02, image_bgr_masked02)
    cv2.imwrite(save_path03, image_bgr_masked03)
    cv2.imwrite(save_path04, image_bgr_masked04)

    image_add0=np.hstack((image_bgr_masked01,image_bgr_masked02))
    image_add1=np.hstack((image_bgr_masked03,image_bgr_masked04))
    image_add=np.vstack((image_add0,image_add1))

    cv2.imwrite(save_path05, image_add)


########################Comment for "Data preparation" code
# 1.(Data preparation 코드 참조)
#   Data 전처리를 사용한다.
#   #(Module을 install해야 하며,Dataset을 다운받아야 합니다.)

# 2.(G.S Train & Validation코드 참조)
#    G.S(Zevin) & Deep Multi-View Models(Sara)모델을 만든후, 전처리 된 Data를 이용하여 Train과 Validataion을 check 합니다.
##   (코드를 사용하기 전에 파이토치에서 Pytorch에서 Gpu사용을 위한 설정과
##   전처리 된 Data의 path를 정확하게 설정하고 Single과 Merged에 따라 net의 구조가 달라짐을 유의합니다. )

# 3.(G.s Test & confusion matrix 코드 참조)
#   2에서 학습된 saved model을 가져와 test set에 대해 학습시키고, confusion matrix를 그립니다.
##   (이때,model의 변수가 동일한지와 save된 train data가 잘 맞는지 확인합니다.)



#