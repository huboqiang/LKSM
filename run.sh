export PYTHONPATH=~/Downloads/hubq/caffe-segnet/python/:$PYTHONPATH


python main.py \
		/home/xufq/Downloads/hubq/CargoKeepLane/dataUsed/TSD-LKSM/${1} \
		/home/xufq/Downloads/hubq/CargoKeepLane/dataUsed/TSD-LKSM-Info/${1}-Info.xml \
		../SegNet-Tutorial/Example_Models/segnet_model_driving_webdemo.prototxt \
		../SegNet-Tutorial/Example_Models/segnet_weights_driving_webdemo.caffemodel 
