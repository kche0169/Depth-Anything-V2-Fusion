import os
import cv2
from depth_anything_v2.dpt import DepthAnythingV2

def main():
    img_path = 'assets/examples/demo01.jpg'
    if not os.path.exists(img_path):
        imgs = [p for p in os.listdir('assets/examples') if p.lower().endswith('.jpg') or p.lower().endswith('.png')]
        if not imgs:
            print('No example images found in assets/examples')
            return
        img_path = os.path.join('assets/examples', imgs[0])

    model = DepthAnythingV2()
    model.eval()
    img = cv2.imread(img_path)
    depth = model.infer_image(img)
    print('Depth shape:', depth.shape)

if __name__ == '__main__':
    main()
