# Pytorch_C3D_Feature_Extractor

pre-trained model (on sport1M) is available:

[C3D_sport.pkl](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle)

### input: video
```
python3 feature_extractor_vid.py -l 6 -i ./data/videos/ -o ./data/c3d_features --OUTPUT_NAME c3d_fc6_features.txt
```

### input: large video

```
python3 feature_extractor_large_vid.py -l 6 -i ./data/videos/ -o ./data/c3d_features --OUTPUT_NAME c3d_fc6_features.txt --BATCH_SIZE 3
```

### reference

https://github.com/yyuanad/Pytorch_C3D_Feature_Extractor