import torch
from bidding_train_env.baseline.dd.DFUSER import (DFUSER)
import time
from bidding_train_env.baseline.dd.dataset import aigb_dataset
from torch.utils.data import DataLoader


def run_decision_diffuser(
        save_path="saved_model/DDtest",
        train_epoch=20,
        batch_size=500):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("train_epoch", train_epoch)
    print("batch-size", batch_size)

    algorithm = DFUSER()
    algorithm = algorithm.to(device)

    args_dict = {'data_version': 'monk_data_small'}
    dataset = aigb_dataset(algorithm.step_len, **args_dict)
    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, num_workers=4, pin_memory=True)

    # 参数数量
    total_params = sum(p.numel() for p in algorithm.parameters())
    print(f"参数数量：{total_params}")

    # 3. 迭代训练

    epi = 1
    for epoch in range(0, train_epoch):
        for batch_index, (states, actions, returns, masks) in enumerate(dataloader):
            states.to(device)
            actions.to(device)
            returns.to(device)
            masks.to(device)

            start_time = time.time()

            # 训练
            all_loss, (diffuse_loss, inv_loss) = algorithm.trainStep(states, actions, returns, masks)
            all_loss = all_loss.detach().clone()
            diffuse_loss = diffuse_loss.detach().clone()
            inv_loss = inv_loss.detach().clone()
            end_time = time.time()
            print(
                f"第{epi}个batch训练时间为: {end_time - start_time} s, all_loss: {all_loss}, diffuse_loss: {diffuse_loss}, inv_loss: {inv_loss}")
            epi += 1

    # algorithm.save_model(save_path, epi)
    algorithm.save_net(save_path, epi)


if __name__ == '__main__':
    run_decision_diffuser()
