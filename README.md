# jai - Just Assemble It!

Author: Jia Geng

Email: jxg570@miami.edu | gjia0214@hotmail.com

## Introduction

*Deep learning is fun. What not fun is the pipeline digging and rigging. Why can't we just enjoy the process
 of exploring all kinds of SOTA techniques with interesting dataset instead of wasting our coffee on boring things like implementing the sockets for the it.*

**jai** is a LEGO-style PyTorch-based Deep Learning Library. 
The main idea behind **jai** is to reduce the amount of time spent on building all sort of pipelines or sockets to plugin those fancy deep learning tricks. This project also tend to create some handy toolkits for Kaggle.

## Dev. Plan

Implement anything popped up in my head when I got time and coffee...

## Library Walk Through 

**Currently the library is in 0.0.-1 version**...

`jai.dataset.py` provides abstract dataset classes that inherit the PyTorch DataSet class. 
The difference is that jai.dataset supports data augmentation and processing.

`jai.improc.py` provides some handy image processing functions, which can be injected to the jai dataset as image
 preprocessing functions or to the augmentation classes as data augmentation functions. 

`jai.augments.py` provides the augmentation classes that can be attached to the jai dataset classes. 
It (will) also provide implementations of some advanced augmentation techniques.

`jai.trainer.py` provide a trainer class that supports classic PyTorch style deep learning training pipeline.
It has some specific requirements on the implementation of the dataset object. 

`jai.logger.py` provides the result/performance logger classes. 
These logger classes can be attached to the trainer during the training stage and able to export, report all kinds of
 model performance related metrics.

`jai.arch.py` (will) provide handy way to modify popular and vanilla deep learning architectures to make the
 architecture compatible with the jai framework.

`jai.kaggler` (will) provides data pipelining solutions, toolbox for general or selected Kaggle project development.
It will also collect some useful tools/models from the kagglers.
 

## Things Need to be Prepared before Use (not fully tested)

0. **Learn how to use `partial()` as it is crucial for this library.**

    ```
    from functiontools import partial
    ```

1. **Prepare/Implement you architecture and loss function.** Some examples can be found in `jai.kaggler.from_kagglers`.
Both need to be in `torch.nn.Module` style. 
If you only need to use the vanilla architectures, just grab a model from the `torchvision.models` and loss function
 from `torch`. 
    E.g.
    ```
    import torchvision.models as model
    import torch.nn as nn
   
    arch = model.resnet18()
    loss = nn.CrossEntropyLoss()
    ```

2. **Implement the dataset class.** Some examples can be found in `jai.kaggler.kaggle_data`. 
The key thing is to inherit the `jai.dataset.JaiDataset`
class and include the following code at the end of the `__getitem()__` method.
The `JaiDataset` constructor can receive two args for preprocessing and augmentation: `tsfms=` `augments=`

    ```
    # do whatever necessary to get the input, and ground truth with input idx
    # img_id is not necessary. but if you have it, the logger will be able to collect false classification during
    evaluation
    # img and t need to be converted to Tensor in correct dimensions
    # img dim: CxHxW; y dim: Bx1 (single output) or BxK (multiple output if you need to predict different things) 
    
    (whatever you implemented) ...
    -> img_id, img, y  
   
    # prepocess the image 
    img = self.prepro(img)
    
    # augment the image during training time
    img = self.augment(img)
    
    # The output need to be dictionary as follow
    # id can be omit
    return {"id": img_id, "x": img, "y": y}
    ```

3. **Prepare preprocessing and augmentation.**
For preprocessing, just use a list to wrap the functions from `jai.improc`. 
The list must contain the `to_tensor` method at the end. 
The wrapped elements must be functions not the function calls. 
Most functions only takes an image input. 
For some functions that takes hyper-parameters, you need to use `paritial(func)` to specify the hyper-parameters.

    E.g.
    ```
    from jai.improc import * 
   
    tsfms = [denoise, partial(threshold, low=15, adaptive_ksize=(13, 13), C=-10), centralize_object, 
             rescale, standardize, to_tensor]
    ```

    For augmentation, create a `jai.augments.FuncAugmentator` object. 
    The `FuncAugmentator` takes a starting probability and a max probability for applying augmentation during training
     time.
    It also takes an augmentation function that process the image. 
    The `func=` also only takes in function instead of function call. 
    And the function should only have one required arg, i.e., the input data. 
    Use `partial()` to wrap the hyper-parameter. `jai.augments.AugF` (will) provide some advanced augmentation.
    E.g:
    ```
    from jai.augments import * 
   
    gridmask = FuncAugmentator(p_start=0.1, p_end=1, func=partial(AugF.grid_mask, d1=96, d2=244))
    ```

4. **Prepare the optimizer and scheduler.** The easiest way is just to grab the optimizer and scheduler from PyTorch. 
You can also implement your own. But make sure use the PyTorch style. Same, use the partial function!
    
    E.g.
    ```
    from torch.optim.adamw import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
   
    optimizer = partial(AdamW, betas=(0.9, 0.999))
    scheduler = partial(CosineAnnealingLR, T_max=100)
    ```

5. **Prepare the `jai.dataset.DataClassDict`.** This is for the purpose of generating logs accordingly.
   E.g. if your model is trying to predict the type of dog in image.
    - `names=` is for hashing the predictors
    - `n_classes=` is for indicating how many possible classes for each predictor.
   
   ```
   from jai.dataset import *
   
   # say your training data have 10 types of dog 
   class_dict = DataClassDict(names=['dog_type'], n_classes=[10])
   ``` 

6. **Prepare the Logger.** You need to prepare a clean directory for receiving log files, 
a prefix string for identifying your trial, and a `ClassDict` for specifying your encoding.
`keep='one_best` and it will only export the best model and overwrite. 
`keep='all_best'` will export all encountered best models.
    
    E.g.
    ```
    from jai.logger import *
    
   # keep
    logger = BasicLogger(log_dst, prefix, class_dict, keep='one_best')
    ```

## Just Assemble It!

Now we have all we need. Next is just assemble it!

We have
```
# model
model = model.resnet18()
loss = nn.CrossEntropyLoss()

# dataset
tsfms = [denoise, partial(threshold, low=15, adaptive_ksize=(13, 13), C=-10), centralize_object, rescale, standardize, to_tensor]
gridmask = FuncAugmentator(p_start=0.1, p_end=1, func=partial(AugF.grid_mask, d1=96, d2=244))
dataset = YourJaiDataset(*args, tsfms=tsfms, augments=gridmask)

# optimzer
optimizer = partial(AdamW, betas=(0.9, 0.999))
scheduler = partial(CosineAnnealingLR, T_max=100)

# logger and predictor encoder
class_dict = DataClassDict(names=['dog_type'], n_classes=[10])
logger = BasicLogger(log_dst, prefix, class_dict, keep='one_best')
```

To Train Your Model:

```
from jai.trainer import *

train_set, eval_set = dataset.split(train_ratio=0.8)  # split to 0.8 : 0.2
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_Set, batch_size=32, shuffle=False)
trainer = BasicTrainer(model, optimizer, scheduler)

trainer.initialize()

trainer.train(train_loader, eval_loader, epochs=30, loss_func=loss, logger=logger)
```

Now you are:
- training your deep learning model with AdamW and CosineAnnealing Scheduler
- using image preprocessing and the GridMask augmentation
- searching for the best model based on the evaluation performance
- recording and exporting the training logs such as
    - batch loss
    - epoch loss and model train/eval accuracy    
    - confusion matrix of your best model(s)
    - export model parameters and optimizer & scheduler state when find better model
    - export the best model's failed detection during eval phase 

After the training is done. You can do: `logger.plot('loss')` to check your training progress.









