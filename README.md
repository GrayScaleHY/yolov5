
1.运行
```
PYTHONPATH=. BOX_SCORE=True python3 models/export.py --weight weights/yolov5l.pt 
```
生成`weights/yolov5l.onnx`

2.将ONNX模型转换到TensorRT模型(在Docker `zldrobit/tensorrt:20.03-fix-resize`)
```
python3 trt_convert.py --onnx=weights/yolov5l.onnx -o weights/yolov5l_nms.engine --explicit-batch --nms [--fp16]

```


3.测试TensorRT的检测结果(在Docker `zldrobit/tensorrt:20.03-fix-resize`)
```
python3 detect.py --weight weights/yolov5l_nms.engine --no-nms
```

4.测试TensorRT模型在TensorRT Server上的预测结果（在Docker `zldrobit/tensorrtserver:20.02-py3-dynamic-batch-size-batchedNMSPlugin`）
在TensorRT Server部署好生成的*.engine文件，例如：重命名为model.plan
```
python3 trt_client_image_v1_trt.py
```
