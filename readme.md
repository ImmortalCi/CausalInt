1. 下载ALI-CCP数据集，解压后放入/data,共有common_features_test.csv  common_features_train.csv  sample_skeleton_test.csv sample_skeleton_train.csv 4个文件
2. 进行数据处理
2.1 运行data_process_new.py，user_f、sample_f、merge_f为文件路径
2.2 运行encode.py，对各域特征进行重编码
2.3 运行split_dataset_new.py,将test set按场景切分
3 运行run.sh,进行训练（注意，test_file路径为/data下的通用格式，列如/data下共有test_domain1.csv、test_domain2.csv、test_domain3.csv，则test_file为test_domain.csv）

1）由于最终结果要多seed取均值，run.sh有循环部分，初次验证时可取单seed实验
2）/data下有处理好的小数据集，可用于训练测试