import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Normal
from tqdm import tqdm
import numpy as np
import time

np.set_printoptions(threshold=np.inf)  # 输出完整张量

mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE)  # 切换动态图模式


# mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE)
class MLP(nn.Cell):
    def __init__(self, num_list=None):
        super(MLP, self).__init__()
        if num_list is None:
            num_list = []
        self.linears = nn.CellList([nn.Dense(num_list[i], num_list[i + 1]) for i in range(len(num_list) - 1)])
        self.bns = nn.CellList([nn.BatchNorm1d(num_list[i + 1]) for i in range(len(num_list) - 1)])
        self.dropout = nn.Dropout(keep_prob=0.9)
        self.activation = nn.LeakyReLU(0.2)

    def construct(self, x):
        # if len(self.linears)==3 and not self.training:
        #     print(x[0][:5])
        for i in range(len(self.linears)):
            x = self.activation(x)
            x = self.linears[i](x)
            # if len(self.linears)==3 and not self.training:
            #     print(x[0][:5])
            # if len(self.linears)==3 and not self.training:
            #     print(x[0][:5])
            if x.shape[0] > 1:
                x = self.bns[i](x)
            x = self.dropout(x)
        # if len(self.linears)==3 and not self.training:
        #     print("\n\n")
        # if len(self.linears)==3 and self.training:
        #     print(self.linears[0].weight[0][:5])
        return x.astype(mindspore.float32)


class Net_loss_1(nn.Cell):
    def __init__(self, extractor, classifier, loss_func):
        super(Net_loss_1, self).__init__()
        self.extractor = extractor
        self.loss_func = loss_func
        self.classifier = classifier

    def construct(self, X, labels):
        labels = labels.view(-1, 1).astype(mindspore.float32)
        X = self.extractor(X)
        logits = self.classifier(X)
        loss = self.loss_func(logits, labels)
        return loss


class Net_loss_2(nn.Cell):  # 前向传播的时候，scenario需是广播后的
    def __init__(self, extractor, classifier, loss_func):
        super(Net_loss_2, self).__init__()
        self.extractor = extractor
        self.loss_func = loss_func
        self.classifier = classifier

    def construct(self, X, scenarios_broadcast):
        X = self.extractor(X)
        logits = self.classifier(X)
        loss = self.loss_func(logits.view(-1, logits.shape[-1]).astype(mindspore.float32),
                              scenarios_broadcast)
        return loss


class Net_loss_1_meta_learning(nn.Cell):
    """
    相比Net_loss_1，区别在于输入即为embedding之后的结果，不需要进行feature extract操作
    """

    def __init__(self, classifier, loss_func):
        super(Net_loss_1_meta_learning, self).__init__()
        self.loss_func = loss_func
        self.classifier = classifier

    def construct(self, X, labels):
        logits = self.classifier(X)
        labels = labels.view(-1, 1).astype(mindspore.float32)
        loss = self.loss_func(logits, labels)
        return loss


class Net_loss_temp(nn.Cell):
    def __init__(self, loss_func, classifier, fuser):
        super(Net_loss_temp, self).__init__()
        # self.Scenario_Representation_index = Scenario_Representation_index
        # self.labels = labels
        # self.index = index
        # self.scenario = scenario
        # self.scenarios = scenarios
        #
        # self.probs_index = probs.take(indices=index, axis=0)
        self.loss_fct = loss_func
        self.classifier = classifier
        self.fuser = fuser

    def construct(self, X, Scenario_Representation_index, labels, index, scenario, scenarios, probs):
        probs_index = probs.take(indices=index, axis=0)
        representation = self.fuser(scenario, Scenario_Representation_index, X, probs_index)
        logits = self.classifier(representation)
        loss_fct = self.loss_fct
        temp_loss = loss_fct(logits,
                             labels.take(indices=index, axis=0).view(-1, 1).astype(mindspore.float32)) * (
                            index.shape[0] / scenarios.shape[0])

        return temp_loss



class Causallnt(nn.Cell):
    def __init__(self):
        super(Causallnt, self).__init__()
        self.feature_fields = 14
        self.embedding_size = 20
        self.num_of_scenario = 3
        self.feature_names = ["101", "121", "122", "124", "125", "126", "127", "128", "129", "205", "206", "207", "210",
                              "216"]
        self.invariant_feature_extractor = MLP(num_list=[280, 256, 128, 64])
        self.scenario_feature_extractor = MLP(num_list=[280, 256, 128, 64])
        self.general_classifier = nn.Dense(64, 1)
        self.scenario_classifier = nn.Dense(64, 3)
        # self.scenario_classifier_1 = nn.Dense(16, 1)
        # self.scenario_classifier_2 = nn.Dense(16, 1)
        # self.scenario_classifier_3 = nn.Dense(16, 1)
        self.scenario_classifier_d = nn.CellList(
            [MLP(num_list=[16, 1]) for i in range(self.num_of_scenario)])  # 这一块不能直接用Dense替换
        # self.scenario_classifier_d = nn.CellList(
        #     [nn.Dense(16, 1) for i in range(self.num_of_scenario)])
        self.scenario_feature_extractor_p = nn.CellList([MLP(num_list=[64, 256]) for i in range(self.num_of_scenario)])
        self.general_classifier_concatenated = nn.Dense(64 * 2, 1)
        self.embedding_length = {"101": 219255, "121": 99, "122": 15, "124": 4, "125": 9, "126": 5, "127": 5, "128": 4,
                                 "129": 6, "205": 310067, "206": 6462, "207": 207638, "210": 92064, "216": 87915}
        self.embedding_list = nn.CellList()  ####!!!!!!!!!!!!!
        self.embedding_dict = {}
        self.w = MLP(num_list=(64, 16))
        self.ratio = 1
        self.__init_weight()

        # 求取grad的写法会报错
        # self.cal_grad_1 = ops.grad(self.calculate_loss_1)
        self.cal_grad_temp = ops.grad(self.calculate_temp_loss, grad_position=0)

        # self.batch_norm=torch.nn.BatchNorm1d(256)
        # self.dropout=torch.nn.Dropout(p=0.1)

        # 尝试提前实例化所有子网络，避免参数重名
        self.net_loss_list = nn.CellList()
        net_loss_1 = Net_loss_1(self.invariant_feature_extractor, self.general_classifier,
                                nn.BCEWithLogitsLoss())
        net_loss_2 = Net_loss_2(self.scenario_feature_extractor, self.scenario_classifier,
                                nn.SoftmaxCrossEntropyWithLogits(reduction="mean"))
        net_loss_1_meta_learning = Net_loss_1_meta_learning(self.general_classifier, nn.BCEWithLogitsLoss())
        self.net_loss_list.append(net_loss_1)
        self.net_loss_list.append(net_loss_2)
        self.net_loss_list.append(net_loss_1_meta_learning)
        # 针对三个scenario_classifier_d， 分别实例化
        self.net_loss_temp_list = nn.CellList()
        net_loss_tmp_1 = Net_loss_temp(nn.BCEWithLogitsLoss(), self.scenario_classifier_d[0], self.fuse_representation)
        net_loss_tmp_2 = Net_loss_temp(nn.BCEWithLogitsLoss(), self.scenario_classifier_d[1], self.fuse_representation)
        net_loss_tmp_3 = Net_loss_temp(nn.BCEWithLogitsLoss(), self.scenario_classifier_d[2], self.fuse_representation)
        self.net_loss_temp_list.append(net_loss_tmp_1)
        self.net_loss_temp_list.append(net_loss_tmp_2)
        self.net_loss_temp_list.append(net_loss_tmp_3)

    def __init_weight(self, ):  # mindspore中没有ModuleDict容器，使用CellList和Dict替换。
        cnt = 0
        for name, size in self.embedding_length.items():
            emb = nn.Embedding(size, self.embedding_size, embedding_table=initializer(
                Normal(sigma=0.01, mean=0.0), shape=[size, self.embedding_size]))
            self.embedding_list.append(emb)
            self.embedding_dict[name] = cnt
            cnt += 1

    def get_Invariant_Representation(self, x):
        leaky_relu = nn.LeakyReLU(0.2)
        return leaky_relu(self.invariant_feature_extractor(x))

    def get_Scenario_Representation(self, x):
        leaky_relu = nn.LeakyReLU(0.2)
        return leaky_relu(self.scenario_feature_extractor(x))

    def test_loss_1(self, inputs, labels):
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=mindspore.Tensor([self.ratio], dtype=mindspore.float32))
        logits = self.general_classifier(inputs)
        loss = loss_fct(logits, labels.view(-1, 1).astype(mindspore.float32))
        return loss

    def calculate_loss_1(self, Invariant_Representation, labels, contxt):
        if contxt == 1:  # 代表需要求导（也就是输入的不是IR，而是原始的input）
            Invariant_Representation = self.invariant_feature_extractor(Invariant_Representation)

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=mindspore.Tensor([self.ratio], dtype=mindspore.float32))

        logits = self.general_classifier(Invariant_Representation)

        if self.training:
            loss = loss_fct(logits, labels.view(-1, 1).astype(mindspore.float32))
            return loss
        else:
            return logits

    def calculate_loss_2(self, Scenario_Representation, scenarios, contxt, batch_size):
        if contxt == 1:  # 代表需要求导（也就是输入的不是SR，而是原始的input）
            Scenario_Representation = self.scenario_feature_extractor(Scenario_Representation)
        softmax = nn.Softmax(axis=-1)
        logits = self.scenario_classifier(Scenario_Representation)
        probs = softmax(logits)
        if self.training:
            loss_fct = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")

            shape = (batch_size, 3)
            broadcast_to = ops.BroadcastTo(shape)
            scenarios = broadcast_to(scenarios.view(-1, 1).astype(mindspore.float32))

            loss = loss_fct(logits.view(-1, logits.shape[-1]).astype(mindspore.float32),
                            scenarios)  # 我改了一下，argmax去掉了/第二次修改，强制把scenarios转换为2维（估计还是不自动广播带来的问题）
            return loss, probs
        else:
            return probs

    def calculate_loss_3(self, Invariant_Representation, Scenario_Representation, labels, scenarios):
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=mindspore.Tensor([self.ratio], dtype=mindspore.float32))
        representation = ops.Concat(axis=-1)((Invariant_Representation, Scenario_Representation))
        logits = self.general_classifier_concatenated(representation)
        loss = loss_fct(logits, labels.view(-1, 1).astype(mindspore.float32))

        return loss

    def calculate_loss_orth(self, inputs, labels, scenarios_broadcast, contxt, batch_size):
        loss = mindspore.Tensor(0., dtype=mindspore.float32)

        grads_1_norm = self.get_grad_on_embeddings((inputs, labels), 1, 1)
        grads_2_norm = self.get_grad_on_embeddings((inputs, scenarios_broadcast), 1, 2)

        # # 再次尝试修改求导；
        # cal_grad_1 = ops.grad(self.calculate_loss_1)
        # cal_grad_2 = ops.grad(self.calculate_loss_2)
        # grad_1 = cal_grad_1(inputs, labels, 1)
        # grad_2 = cal_grad_2(inputs, scenarios, 1, batch_size)
        # grads_1_norm = self.get_grad_on_embeddings(grad_1)
        # grads_2_norm = self.get_grad_on_embeddings(grad_2)

        loss = ops.ReduceSum()(grads_1_norm * grads_2_norm, 1)
        loss = ops.Square()(loss).mean()  ##ops.Square()
        return loss

    def fuse_representation(self, d, Scenario_Representation, Invariant_Representation, probs):
        leaky_relu = nn.LeakyReLU(alpha=0.2)
        # print(1)
        H = []
        Expert = []
        # print(1.1)
        for p in range(self.num_of_scenario):
            h = leaky_relu(self.scenario_feature_extractor_p[p](Scenario_Representation))
            H.append(h)
            if d == p:
                Expert.append(H[p])
            else:
                Expert.append(H[p].copy())
        TransNet = ops.ZerosLike()(Expert[0])
        for p in range(self.num_of_scenario):
            TransNet = TransNet + probs[:, p:p + 1] * Expert[p]
        X = self.w(Invariant_Representation)
        return ops.ReduceSum()(ops.ExpandDims()(X, -1) * TransNet.view(-1, 16, 16), 1)

    def calculate(self, Invariant_Representation_index, Scenario_Representation_index, labels, probs, index, scenario,
                  scenarios):
        probs_index = probs.take(index, 0)  # self.index_select(probs,0,index)   ####!!!!!
        representation = self.fuse_representation(scenario, Scenario_Representation_index,
                                                  Invariant_Representation_index, probs_index)
        if scenario == 0:
            logits = self.scenario_classifier_1(representation)
        elif scenario == 1:
            logits = self.scenario_classifier_2(representation)
        else:
            logits = self.scenario_classifier_3(representation)
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=mindspore.Tensor([self.ratio], dtype=mindspore.float32))
        temp_loss = loss_fct(logits, labels.take(indices=index, axis=0).view(-1, 1).astype(mindspore.float32)) * (
                index.shape[0] / scenarios.shape[0])
        return temp_loss

    def calculate_temp_loss(self, Invariant_Representation_index, Scenario_Representation_index, labels, index,
                            scenario, scenarios, probs):
        probs_index = probs.take(indices=index, axis=0)
        representation = self.fuse_representation(scenario, Scenario_Representation_index,
                                                  Invariant_Representation_index, probs_index)
        logits = self.scenario_classifier_d[scenario](representation)
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=mindspore.Tensor([self.ratio], dtype=mindspore.float32))
        temp_loss = loss_fct(logits, labels.take(indices=index, axis=0).view(-1, 1).astype(mindspore.float32)) * (
                index.shape[0] / scenarios.shape[0])
        return temp_loss

    def calculate_loss_d(self, Invariant_Representation, Scenario_Representation, labels, scenarios, probs):
        loss_list = []
        grad_list = []
        all_logits = mindspore.Tensor([], dtype=mindspore.float32)
        all_index = mindspore.Tensor([], dtype=mindspore.float32)
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=mindspore.Tensor(np.array([self.ratio]), dtype=mindspore.float32))  #
        loss = mindspore.Tensor(0., dtype=mindspore.float32)
        # print('step1```````````````````````````````````````````finished')
        for i in range(self.num_of_scenario):
            scenario = i

            # True->1; False->0
            # 必须得转换成numpy再转回来，否则take操作会报[fill]操作不支持，目前还没发现原因。
            # index_numpy = ops.nonzero(scenarios == scenario).view(-1).asnumpy()  # 1.7不支持这个操作，如下⬇️等价
            # index = mindspore.Tensor(index_numpy)
            index = ops.nonzero(scenarios == scenario).view(-1).asnumpy()
            index = mindspore.Tensor(index)
            # print('场景为', scenario)
            # print('样本数', index.shape)

            # print('step3````````````````````````````````````````````finished')
            if index.shape[0] == 0:  # 当前batch没有属于场景d的样本
                if self.training:
                    temp_loss = mindspore.Tensor(0., dtype=mindspore.float32)  #####!!!!!
                    loss_list.append(temp_loss)
                    grad_list.append(None)
                    loss = loss + temp_loss
            else:
                Scenario_Representation_index = Scenario_Representation.take(index, 0)
                Invariant_Representation_index = Invariant_Representation.take(index, 0)
                probs_index = probs.take(index, 0)
                representation = self.fuse_representation(scenario, Scenario_Representation_index,
                                                          Invariant_Representation_index, probs_index)
                logits = self.scenario_classifier_d[i](representation)

                if self.training:
                    temp_loss = loss_fct(logits,
                                             labels.take(index, 0).view(-1, 1).astype(mindspore.float32)) * (
                                            index.shape[0] / scenarios.shape[0])

                    # 尝试看logits、labels信息
                    # print('logits:', logits)
                    # print('labels:', labels)
                    # print('labels selected', labels.take(index, 0).view(-1, 1).astype(mindspore.float32))
                    loss_list.append(temp_loss)

                    # # 求取grad，放入grad_list中

                    # grad = self.calculate_grad((Invariant_Representation_index, Scenario_Representation_index,
                    #                             labels, index, scenario, scenarios, probs), 2, scenario + 1).sum(axis=0,
                    #                                                                                              keepdims=True)
                    # grad = self.cal_grad_temp(Invariant_Representation_index, Scenario_Representation_index,
                    #                           labels, index, scenario, scenarios, probs).sum(axis=0, keepdims=True)
                    grad_list.append(1)


                    print(temp_loss)
                    loss = loss + temp_loss
                else:
                    all_logits = ops.Concat(axis=0)((all_logits, logits.view(-1)))
                    all_index = ops.Concat(axis=0)((all_index, index.view(-1).astype(mindspore.float32)))
        if self.training:
            # return loss, loss_list, grad_list  # 暂时停掉
            return loss
        else:
            return all_logits.take(indices=ops.sort(all_index)[1],
                                   axis=0)
            # return self.scenario_classifier_d[2].weight.asnumpy(), self.scenario_classifier_d[2].bias.asnumpy(), representation, logits

    def calculate_loss_orth_all(self, loss_list, Invariant_Representation, inputs, scenarios, grad_list):
        def proj(m, n):
            ratio = ops.ReduceSum()(m * n, axis=1) / (ops.ReduceSum()(m * m, axis=1) + 1e-9)
            return ratio.view(-1, 1) * m

        g = []
        b = []
        for i in range(self.num_of_scenario):
            g.append(grad_list[i])
        for i in range(self.num_of_scenario):
            if g[i] is None:
                b.append(None)
                continue
            temp = g[i]
            for j in range(i):
                if b[j] is None:
                    continue
                temp = temp - proj(b[j], g[i])
            b.append(temp)
        loss = mindspore.Tensor(0., dtype=mindspore.float32)  #######!!!!!!
        for i in range(self.num_of_scenario):
            if g[i] is None:
                continue
            loss = loss + ((g[i] - b[i]) ** 2).sum() / 2
        return loss

    def get_grad_on_embeddings(self, inputs, lst_num, loss_num):
        grads = self.calculate_grad(inputs, lst_num, loss_num)
        return ops.L2Normalize(axis=1)(grads)  # 默认轴不一样，mindspore默认是0，pytorch默认是1，注意！！
    # def get_grad_on_embeddings(self, grad):
    #     return ops.L2Normalize(axis=1)(grad)

    def calculate_grad(self, inputs, lst_num, loss_num):
        """

        :param loss_net:用于一次前向传播的抽离网络，该网络的前向传播由loss的计算公式替代
        :param all_params:  主要包含两部分：网络构建的参数、网络的输入. loss1:Tuple[输入、编码、分类器、损失函数、标签]; loss2:Tuple[输入、编码、分类器、损失函数、场景]
        :param loss_num: 1:Tuple[输入、编码、分类器、损失函数、标签];
                         2:Tuple[输入、编码、分类器、损失函数、场景、batch_size] ；
                         3:Tuple[输入、分类器、损失函数、标签]
                         4:Tuple[IR_index、SR_index、标签labels、索引index、当前场景scenario、场景数scenarios、概率probs、融合器、分类器、损失函数loss_fuc]
        :return: 对第一个输入的梯度
        """
        # origi_input = all_params[0]
        # if loss_num == 1:
        #     net = loss_net(all_params[1], all_params[2], all_params[3], all_params[4])
        # elif loss_num == 2:
        #     net = loss_net(all_params[1], all_params[2], all_params[3], all_params[4], all_params[5])
        # elif loss_num == 3:  # for meta learning
        #     net = loss_net(all_params[1], all_params[2], all_params[3])
        # else:  # for temp_loss
        #     net = loss_net(all_params[1], all_params[2], all_params[3], all_params[4], all_params[5], all_params[6],
        #                    all_params[7], all_params[8], all_params[9])

        # Grad_op = ops.GradOperation()
        # Grad_func = Grad_op(net)
        # grad = Grad_func(origi_input)

        # 尝试实例映射

        # if lst_num == 1:
        #     net = self.net_loss_list[loss_num-1]
        # else:
        #     net = self.net_loss_temp_list[loss_num-1]
        #
        # Grad_op = ops.GradOperation()
        # Grad_func = Grad_op(net)
        # grad = Grad_func(*inputs)

        # 实例化遇到爆显存问题，尝试不实例化
        Grad_op = ops.GradOperation()
        if lst_num == 1:
            Grad_func = Grad_op(self.net_loss_list[loss_num - 1])
            grad = Grad_func(*inputs)
        else:
            Grad_func = Grad_op(self.net_loss_temp_list[loss_num - 1])
            grad = Grad_func(*inputs)

        return grad

    def construct(self, batch):
        for name in self.feature_names:
            batch[name] = batch[name].squeeze()
        inputs = None
        labels = batch["click"].astype(mindspore.float32)
        batch_size = labels.shape[0]
        scenarios = batch["301"] - 1
        scenarios = scenarios.astype(mindspore.float32)

        # 广播好的scenarios
        shape = (batch_size, 3)
        broadcast_to = ops.BroadcastTo(shape)
        scenarios_broadcast = broadcast_to(scenarios.view(-1, 1).astype(mindspore.float32))

        for name in self.feature_names:
            # print("ok17!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if name == "210":
                # print("ok18!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                x = batch[name].copy()
                is_pad = (x == -1)
                is_norm = (x != -1)
                x[is_pad] = 0
                # print(x.shape)
                # print(is_pad.shape)
                idx = self.embedding_dict[name]  # 索引到对应的embedding层
                input = self.embedding_list[idx](x)
                # is_pad = is_pad.view(6000, 40, 1).broadcast_to((6000, 40, 20))# ms不支持自动广播,这里的6000是batch size，ms1.7不支持当前操作，使用下面的平替

                shape = (batch_size, 40, 20)
                broadcast_to = ops.BroadcastTo(shape)
                is_pad = broadcast_to(is_pad.view(batch_size, 40, 1))

                input[is_pad] = 0

                input = input.astype(mindspore.float32)
                is_norm = is_norm.astype(mindspore.float32)

                # input = ops.ReduceSum()(input, axis=1) / ops.ReduceSum()(is_norm, axis=-1).view(-1, 1)  索引报错
                reducesum_op = ops.ReduceSum()
                input = reducesum_op(input, 1) / reducesum_op(is_norm, -1).view(-1, 1)

            else:
                idx = self.embedding_dict[name]
                input = self.embedding_list[idx](batch[name])

            if inputs is None:
                inputs = input
            else:
                inputs = ops.Concat(axis=-1)((inputs, input))

        inputs = inputs.astype(mindspore.float32)
        Invariant_Representation = self.invariant_feature_extractor(inputs).astype(mindspore.float32)
        Scenario_Representation = self.scenario_feature_extractor(inputs).astype(mindspore.float32)

        if self.training:
            print("ok20!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            loss_1 = self.calculate_loss_1(Invariant_Representation, labels, 0)
            print("loss_1 finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", loss_1)
            loss_2, probs = self.calculate_loss_2(Scenario_Representation, scenarios, 0, batch_size)
            print("loss_2 finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", loss_2)
            loss_3 = self.calculate_loss_3(Invariant_Representation, Scenario_Representation, labels, scenarios)
            print("loss_3 finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", loss_3)
            # loss_orth = self.calculate_loss_orth(inputs, labels, scenarios_broadcast, 1, batch_size)
            # print("loss_orth finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", loss_orth)



            # Invariant_Representation_ = Invariant_Representation - self.lr * self.calculate_grad((
            #     Invariant_Representation, labels), 1, 3)
            # grad = ops.grad(self.test_loss_1, grad_position=0)(Invariant_Representation, labels)
            # Invariant_Representation_ = Invariant_Representation - self.lr * grad
            #
            # print("Invariant_Representation_finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # loss_d, loss_d_list, grad_list = self.calculate_loss_d(Invariant_Representation_, Scenario_Representation,
            #                                                        labels, scenarios, probs)
            loss_d = self.calculate_loss_d(Invariant_Representation, Scenario_Representation,
                                                                   labels, scenarios, probs)
            print("loss_d,loss_d_list,grad_list finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", loss_d)
            # loss_orth_all = self.calculate_loss_orth_all(loss_d_list, Invariant_Representation, inputs, scenarios,
            #                                              grad_list)
            # print("loss_orth_all finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", loss_orth_all)
            # # loss = loss_d
            # loss = loss_1 + loss_2 + loss_3 + loss_orth + loss_d + loss_orth_all
            loss = loss_1 + loss_2 + loss_3 + loss_d

            print("loss finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", loss)
            return loss
        else:
            probs = self.calculate_loss_2(Scenario_Representation, scenarios, 0, batch_size)
            logits = self.calculate_loss_d(Invariant_Representation, Scenario_Representation, labels, scenarios, probs)
            return logits
            # weight, bias, rep, logits = self.calculate_loss_d(Invariant_Representation, Scenario_Representation, labels, scenarios, probs)
            # return weight, bias, rep, logits
