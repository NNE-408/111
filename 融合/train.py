import os

from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import Adam

from nets.deeplab import Deeplabv3
from nets.deeplab_training import CE, Generator, LossHistory, dice_loss_with_CE
from utils.metrics import Iou_score, f_score

if __name__ == "__main__":     
    log_dir = "logs/"
    #------------------------------#
    #   输入图片的大小
    #------------------------------#
    input_shape = [512,512,3]
    
    
    num_classes = 2
    
    
    dice_loss = False
    
    
    
    backbone = "mobilenet"
    #---------------------#
    #   下采样
    #---------------------#
    downsample_factor = 16
  
  
  
    dataset_path = "VOCdevkit/VOC2007/"


    model = Deeplabv3(num_classes,input_shape,backbone=backbone,downsample_factor=downsample_factor)



    model_path = "model_data/deeplabv3.h5"
    model.load_weights(model_path,by_name=True,skip_mismatch=True)


    with open(os.path.join(dataset_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()


    with open(os.path.join(dataset_path, "ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
        
    #-------------------------------------------------------------------------------#
    #   reduce_lr用于设置学习率下降的方式
    #-------------------------------------------------------------------------------#
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir)
    loss_history = LossHistory(log_dir)

    if backbone=="mobilenet":
        freeze_layers = 146
    else:
        freeze_layers = 358

    for i in range(freeze_layers): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用
    #------------------------------------------------------#
    if True:
        lr              = 1e-4
        Init_Epoch      = 0
        Freeze_Epoch    = 50
        Batch_size      = 8
        
        # 交叉熵
        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])

        gen             = Generator(Batch_size, train_lines, input_shape, num_classes, dataset_path).generate()
        gen_val         = Generator(Batch_size, val_lines, input_shape, num_classes, dataset_path).generate(False)

        epoch_size      = len(train_lines)//Batch_size
        epoch_size_val  = len(val_lines)//Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))
        model.fit_generator(gen,
                steps_per_epoch=epoch_size,
                validation_data=gen_val,
                validation_steps=epoch_size_val,
                epochs=Freeze_Epoch,
                initial_epoch=Init_Epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, loss_history])
    
    for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        lr              = 1e-5
        Freeze_Epoch    = 50
        Unfreeze_Epoch  = 200 
        Batch_size      = 4
        
        # 交叉熵
        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])

        gen             = Generator(Batch_size, train_lines, input_shape, num_classes, dataset_path).generate()
        gen_val         = Generator(Batch_size, val_lines, input_shape, num_classes, dataset_path).generate(False)
        
        epoch_size      = len(train_lines)//Batch_size
        epoch_size_val  = len(val_lines)//Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))
        model.fit_generator(gen,
                steps_per_epoch=epoch_size,
                validation_data=gen_val,
                validation_steps=epoch_size_val,
                epochs=Unfreeze_Epoch,
                initial_epoch=Freeze_Epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, loss_history])
        model.save_weights('logs/last1.h5')
