import numpy as np

import torch

# s_vec_label = convert_to_onehot(s_sca_label, CLASS_NUM)


def convert_to_onehot(sca_label, class_num):
    return np.eye(class_num)[sca_label]

class Weight:

    @staticmethod
    # weight_ss, weight_tt, weight_st = Weight.cal_weight(s_label, t_label, type='visual',batch_size=BATCH_SIZE, class_num=CLASS_NUM)

    def cal_weight(s_label, t_label, batch_size,CLASS_NUM):
        batch_size = s_label.size()[0]

        # # label_list = list(set(s_label.data.cpu().numpy()))
        # #
        # # CLASS_NUM = len(label_list)
        # # print('label list', label_list)
        # # print('class num',CLASS_NUM)
        #
        # CLASS_NUM = int(torch.max(s_label)) + 1

        #计算核函数前的权值（源域）
        s_sca_label = s_label.cpu().data.numpy()

        s_vec_label = convert_to_onehot(s_sca_label,CLASS_NUM)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, CLASS_NUM)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum
        #计算核函数前的权值（目标域）
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        #t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()

        t_sum = np.sum(t_vec_label, axis=0).reshape(1, CLASS_NUM)

        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        count = 0
        for i in range(CLASS_NUM):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                ss = np.dot(s_tvec, s_tvec.T)
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
