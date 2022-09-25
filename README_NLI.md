NLI文件使用说明
finetune：
finetune_nli_for_hallu_debug.py
使用时需要修改的项包括：
dataset_name = "chineseNLI" finetune数据集名称
data_dir = "data/{dataset}".format(dataset=dataset_name) 数据集地址，需要整理成特定形式，可参考data/中数据
save_dir = "cpt/{dataset}".format(dataset=dataset_name) cpt保存地址
save_cpt_name = "{dataset}_{steps}".format(dataset=dataset_name,steps = str(max_steps)) cpt保存文件名
num_labels = 3 NLI分类数量
v_d={
            b"contradiction": 0,
            b"entailment": 1,
            b"neutral": 2,
        }  所有分类的label，及其对应编号（编号可随意顺序，但是finetune与predict的vocal dict需要保证编号顺序相同

predict：
predict.py
测试数据参考data/chineseNLI/predict/pre
使用时需要修改的项包括：
num_label = 3 NLI分类数量
file_name 被评测的文件名称
data_file 被评测数据地址（不包括文件名）
init_checkpoint cpt存储位置 参考文书文档
save_file 输出结果存储位置
inv_vocab_dict={
            0:"contradiction",
            1:"entailment",
            2:"neutral",
        }  与finetune时保持一致

evaluate：
evaluate.py
使用时需要修改的项包括：
num_label = 3 NLI分类数量
file_name 被评测的文件名称
data_file 被评测数据地址（不包括文件名）
init_checkpoint cpt存储位置
save_file 输出结果存储位置
inv_vocab_dict={
            0:"contradiction",
            1:"entailment",
            2:"neutral",
        }  与finetune时保持一致