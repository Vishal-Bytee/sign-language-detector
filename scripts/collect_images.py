import cv2
import os
import time

signs = ['Hello', 'ThankYou', 'Yes', 'No', 'ILoveYou', 'Please', 'Sorry', 'eat','drink']

IMAGES_PER_SIGN = 50


SAVE_PATH = 'collected_images'


cap = cv2.VideoCapture(0)

for sign in signs:
    # Har sign ka alag folder banao
    folder = os.path.join(SAVE_PATH, sign)
    os.makedirs(folder, exist_ok=True)

    print(f'GEt ready !  show Sign : {sign}')
    print('3 second mein shuru hoga...')
    time.sleep(3)

    count = 0
    while count < IMAGES_PER_SIGN:
        ret, frame = cap.read()

    
        cv2.putText(frame,
                    f'{sign}: {count}/{IMAGES_PER_SIGN}',
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow('Taking photos -  press Q for closing', frame)

    
        img_path = os.path.join(folder, f'{sign}_{count}.jpg')
        cv2.imwrite(img_path, frame)
        count += 1
        time.sleep(0.1)

    
        if cv2.waitKey(1) == ord('q'):
            break

    print(f'{sign} ke {count} photos le liye! ✓')

cap.release()
cv2.destroyAllWindows()
print('all photos are taken successfully!')