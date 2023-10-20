
import argparse
from mindspore.dataset import CSVDataset
import numpy as np
# from model_mindspore import Causallnt
from model import Causallnt
import math
from dataclasses import dataclass
import os
import time
from tqdm import trange,tqdm
from sklearn.metrics import precision_recall_curve,auc,roc_auc_score
import mindspore
import mindspore.dataset as ds
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore import ops
from mindspore.ops import GradOperation
from mindspore.nn.optim import Momentum

# 初步
from mindspore.context import ParallelMode
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter
from mindspore import ParameterTuple

import mindspore.ops.functional as F
from mindspore.ops import composite as C
from mindspore import dtype as mstype
from mindspore import context
from mindspore import ms_function
from mindspore.communication.management import get_group_size
from mindspore.parallel._auto_parallel_context import auto_parallel_context


mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE)

class WithGradCell(nn.Cell):
    """train one step cell with sense"""

    def __init__(self, network, optimizer, clip_value=0.1):
        super().__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.scale_sense = Parameter(Tensor(1., dtype=mstype.float32), name="scale_sense")
        self.reducer_flag = False
        self.grad_reducer = None
        self.max_grad_norm = clip_value
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        # this is a hack
        self.enable_tuple_broaden = True

    @ms_function
    def clip_backward(self, loss, grads):
        grads = ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss

    def construct(self, *inputs):
        """construct"""
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs, self.scale_sense)
        return self.clip_backward(loss, grads)


class WarmUpPolynomialDecayLR(LearningRateSchedule):
    """"""
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.end_learning_rate = end_learning_rate
        self.decay_steps = decay_steps
        self.power = power

    def construct(self, global_step):
        # warmup lr
        warmup_percent = global_step.astype(mindspore.float32) / self.warmup_steps
        warmup_learning_rate = self.learning_rate * warmup_percent
        # polynomial lr
        global_step = ops.minimum(global_step, self.decay_steps)
        decayed_learning_rate = (self.learning_rate - self.end_learning_rate) * \
                                ops.pow((1 - global_step / self.decay_steps), self.power) + \
                                self.end_learning_rate
        is_warmup = (global_step < self.warmup_steps).astype(mindspore.float32)
        learning_rate = ((1.0 - is_warmup) * decayed_learning_rate + is_warmup * warmup_learning_rate)
        return learning_rate

def create_optimizer(optim_params, init_lr, num_train_steps, num_warmup_steps, eps=5e-8):
    lr = WarmUpPolynomialDecayLR(init_lr, 0.0, num_warmup_steps, num_train_steps, 1.0)
    optim = mindspore.nn.Adam(optim_params, lr, eps)
    return optim


@dataclass
class DataCollotar:
    max_num_of_210:int
    def __call__(self, features, return_tensors=None):
        result={}
        for feature in features:
            for name,value in feature.items():
                # if name=="210":
                #     value=[int(v) for v in value.split("#")]
                #     value.extend((self.max_num_of_210-len(value))*[-1])
                if name in result:
                    result[name].append(value)
                else:
                    result[name]=[value]
        for key in result.keys():
            result[key]=mindspore.tensor(result[key],dtype=mindspore.int32)
        return result


def set_seed(seed1):
    mindspore.set_seed(seed=seed1)

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("\n\n\n\n")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    print("\n\n\n\n")
    input_columns=["101","121","122","124","125","126","127","128","129","205","206","207","210","216"]

    if args.seed is not None:
        set_seed(args.seed)

    datacollotar=DataCollotar(max_num_of_210=40)

    def get_dataloader(filename, batch_size,shuffle):
        max_num_of_210=40
        column_defaults =[0,1,0,143,12702,'10554#1725#1727#1729#1723#26606#1724#26607',7362,0,1,7,1,5,1,1,1,2]

        def preprocess_function(example):
            tmp_list=[]
            example = str(example)[2:-1]
            for j in example.split("#"):
                if len(j) == 0: continue
                tmp_list.append(int(j))
            tmp_list.extend((max_num_of_210-len(tmp_list))*[-1])
            return np.array(tmp_list)

        dataset = CSVDataset(
            dataset_files=filename,
            field_delim=",",
            column_defaults=column_defaults,
            num_parallel_workers=args.dataloader_num_workers,
            num_samples=None,
            shuffle=shuffle)
        dataset = dataset.map(operations=[(lambda x: preprocess_function(x))], input_columns=["210"])
        dataset = dataset.batch(batch_size)
        return dataset

    train_dataloader=get_dataloader(args.train_file,args.batch_size,shuffle=True)
    test_dataloader_list=[]
    for i in range(1,4):
        test_dataloader_list.append(get_dataloader(args.test_file.replace("domain","domain"+str(i)),args.batch_size,shuffle=False))


    print(train_dataloader.get_dataset_size())
    print('-'*50)
    model=Causallnt()

    # input_tensor = mindspore.Tensor(np.ones((14, 20)).astype(np.float32))
    # # mindspore.context.set_context(save_graphs=True,save_graphs_path="./saved_graphs/",save_graph_dot=True)
    # mindspore.export(model,input_tensor,file_name="obf_net",file_format="MINDIR")
    # exit()

    model.lr=args.learning_rate

    no_decay = ["bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.parameters_and_names() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.parameters_and_names() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    num_update_steps_per_epoch = math.ceil(train_dataloader.get_dataset_size() / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # 改了一下参数？！！这个优化器目前有问题，需要补全。
    # optimizer_adam = create_optimizer(model.trainable_params(), init_lr = args.learning_rate, \
    #                                   num_train_steps = args.max_train_steps * args.gradient_accumulation_steps, \
    #                                   num_warmup_steps = args.num_warmup_steps * args.gradient_accumulation_steps,
    #                                   eps = 5e-8)
    """
    尝试重写优化器，2月26日，临时改成Momentum，测试一下效果
    """
    optimizer = Momentum(params=model.trainable_params(), learning_rate=0.001, momentum=0.9)
    #optimizer = mindspore.nn.Adam(optimizer_grouped_parameters, learning_rate=args.learning_rate, eps=5e-8)
    #print(args.num_warmup_steps * args.gradient_accumulation_steps)
    #lr_scheduler = get_scheduler(
    #    name=args.lr_scheduler_type,
    #    optimizer=optimizer,
    #    num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
    #    num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    #)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 初始化训练模块
    train_module = WithGradCell(model, optimizer)
    def cal_auc(label: list, pred: list):
        auc = roc_auc_score(label, pred)
        return auc



    sigmoid=mindspore.ops.Sigmoid()
    best_test_auc=0
    best_epoch=-1
    log_path=os.path.join(args.output_dir,"result.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    fw=open(log_path,"w")
    for epoch in trange(args.num_train_epochs):
        model.set_train()
        print(model._phase)
        print(f"\nepoch {epoch+1}")
        total_loss=0
        t=tqdm(train_dataloader.create_dict_iterator())
        #print("ok1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(train_dataloader.get_dataset_size())
        #print('-'*50)
        #print('start_run')

        batch_num =0

        for step, batch in enumerate(t):
            #print("batch", batch)
            #print("ok2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            for key in batch.keys():
                #print(key)
                #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                batch[key]=batch[key]
            #print("ok3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            t.set_description(f"epoch {epoch+1} step {step+1}:")
            #print("ok4!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


            # 尝试更换训练写法
            # grad = ops.GradOperation()(model)(batch)
            # import pdb
            # pdb.set_trace()
            # optimizer_adam(grad)

            # 老版本
            # train_one_step = mindspore.nn.TrainOneStepCell(model, optimizer_adam)
            # loss = train_one_step(batch).view(1)

            # 使用新写法

            loss = train_module(batch).view(1)
            # import pdb
            # pdb.set_trace()

            """
            ms只支持单输出loss的网络，精简输出
            """
            # loss, loss_1, loss_2, loss_3, loss_orth, loss_d, loss_orth_all = model.construct(batch)
            # loss = model.construct(batch)
            print("ok6!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            total_loss += loss.item()
            print("ok7!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Grad_op = ops.GradOperation()
            # Grad_func = Grad_op(Causallnt)
            # grads = Grad_func(batch)
            # print("ok9!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # optimizer_adam(grads)
            print("ok10!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            # # 尝试换一种训练写法
            # loss = model(batch)

            # print(loss.size)
            # print(loss.item())
            # exit()
            # t.set_postfix(loss=loss.item(),loss_1=loss_1.item(),loss_2=loss_2.item(), loss_3=loss_3.item(),loss_orth=loss_orth.item(),loss_d=loss_d.item(),loss_orth_all=loss_orth_all.item())
            t.set_postfix(loss=loss.item())

            # 结束一个batch的训练
            batch_num += 1

            print("ok11!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            '''
            loss,loss_1,loss_2,loss_3,loss_orth,loss_d,loss_orth_all=model(batch)
            # loss=model(batch)
            total_loss+=loss.item()
            t.set_postfix(loss=loss.item(),loss_1=loss_1.item(),loss_2=loss_2.item(),loss_3=loss_3.item(),loss_orth=loss_orth.item(),loss_d=loss_d.item(),loss_orth_all=loss_orth_all.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''
        # avg_loss=total_loss/len(train_dataloader) #len 报错
        avg_loss=total_loss/batch_num
        print(f"avg loss in epoch {epoch+1}:{avg_loss}")
        log={}
        log["epoch"]=epoch+1
        log["avg_loss"]=avg_loss
        path=os.path.join(args.output_dir,f"epoch_{epoch+1}")
        if not os.path.exists(path):
            os.makedirs(path)
        mindspore.save_checkpoint(model,path+"/model.pth")


        model.set_train(False)
        test_auc=0
        for t in range(1,4):
            print(f"\n\neval on test domain{t} set")
            labels=[]
            predicts=[]
            negative=0
            positive=0
            true=0
            # for step, batch in enumerate(tqdm(test_dataloader_list[t-1])):  # 报错
            for step, batch in enumerate(tqdm(test_dataloader_list[t-1].create_dict_iterator())):
                for key in batch.keys():
                    batch[key]=batch[key]
                logits=model(batch)
                # import pdb
                # pdb.set_trace()
                # weight, bias, rp, logits = model(batch)
                # print('weights: \n',  weight)
                # print('bias: \n' , bias)
                # print("representations: \n" , rp)
                # print("logits: \n" , logits)
                probs=sigmoid(logits)
                # print("probs: \n" , probs)
                labels.extend(batch["click"].asnumpy())

                predicts.extend(probs.view(-1).asnumpy())
                # print(predicts)
                # print(labels)
                # exit()
            for i in range(len(labels)):
                if predicts[i]<0.5:
                    predict=0
                else:
                    predict=1
                if predict==labels[i]:
                    true+=1
            fout=open("result"+str(t)+".txt","w")
            for i in range(len(labels)):
                fout.write(str(labels[i])+" "+str(predicts[i])+"\n")
            fout.close()
            acc=true/len(labels)
            auc=cal_auc(labels,predicts)
            print("acc:",acc)
            print("AUC:",auc)
            test_auc+=auc
            log[f"test_auc on domain{t}"]=auc
        fw.write(str(log)+"\n")
        test_auc/=3
        if test_auc>best_test_auc:
            best_test_auc=test_auc
            best_epoch=epoch+1
    fw.write(f"best epoch:{best_epoch}")
    fw.close()




if __name__ == "__main__":
    main()
            
