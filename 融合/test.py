


from nets.deeplab import Deeplabv3

if __name__ == "__main__":
    model = Deeplabv3(21, [512,512,3], backbone='mobilenet')


    for i,layer in enumerate(model.layers):
        print(i,layer.name)
