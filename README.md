User-Information-mining

This repository contains a number of different models implemented 
in TensorFlow for user information mining based on sensor data;

project tree

- README.md
- output/
- log/
- model/

	1  recognition/(gender; age; height; weight)
	2  authentication/(multi; binary; index)
	
- data/

    1  HAR
    2  WISDM
    3  HASC
	
- src/utils

	mysqlDAL()					#数据处理层
	mysql_dump()				
	sample()					#采样
	calforroc()					#计算tpr fpr roc曲线
	condict()					#设置条件字典
	
- src/

	class Proxy():				#代理 任务模式切换
		
		setconfig()
			DataConf()			#配置数据处理方式
			ModelConf()			#配置模型参数
		
		apply(config)			#启动配置文件
			DataPreprocess()	#数据预处理

		para_train()			#多模型并行训练
		
	class DataConf():			#数据处理方式配置
	
		setdatasrc()
		setclass()
		settask()
		setDLconf()
		
	class ModelConf():			#模型配置
	
		setbatch_size()
		setlearning_rate()
		setepoch()
		setchannels()
		setsteps()
	
	class DataPreprocess():	
	
		dump_data()  			#判空 mysql_dump()
		para_cut()				#并行切数据调度
		segment()				
		windows()
		standardize()
		labeling()				#标签处理
		
		read_hasc()
		read_har()
		read_wisdm()
	
	class ModelBuilder():	
		
		train_bilstm()			#模型计算图
		train_cnn()
		train_vgg()
		train_deeplstm()
		train_deepgru()
		train_inception()
		train_vgg_lstm()
		
		run_session()
		test()
		get_batch()				
		logger_result()			#结果记录
		
		
		
		
	
	