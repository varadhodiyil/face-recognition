
import cv2
from opencv_utils import OpenCVHelper
from predictor import Predictor

helper = OpenCVHelper()

cap = cv2.VideoCapture(0)
f_verify = False
f_img_verify = False
st_verify = [False, False, False, False, False]
p = Predictor()
if cap.isOpened():
    while True:

        ret, image = cap.read()
        if ret:
            cv2.imshow("Faces found", image)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_face, pos_face = helper.convert_img(image)
            
            # img_face = helper.convert_to_greyscale(image)
            
            if img_face is not None:
                cv2.rectangle(image, (pos_face[0], pos_face[2]), (pos_face[1], pos_face[3]), (255, 0, 0), 5)
                f_img_verify = p.classify_image(img_face)

                # for i in range(4):
                #     st_verify[i] = st_verify[i + 1]

                # st_verify[4] = f_img_verify

                print f_img_verify
                if st_verify == [True, True, True, True, True]:
                    f_verify = True
                elif st_verify == [False, False, False, False, False]:
                    f_verify = False

                cv2.putText(image,f_img_verify , (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Faces found", image)
            else:
                print("No Detection!")
        else:
            continue
        

        
        k = cv2.waitKey(30) & 0xff

        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
