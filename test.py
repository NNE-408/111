
from tensorflow.keras.layers import Input

from nets.yolo4 import yolo_body

if __name__ == "__main__":
    inputs = Input([416,416,3])
    model = yolo_body(inputs,3,80)

