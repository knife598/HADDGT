import copy

import torch
from torch import nn
from utilz.utils import *
from models.modules import *


def metric_errors_from_grid(Pred_xy, Target_xy, config, valid_mask=None):
    bounds = config.get('metric_coord_bounds')
    if not bounds:
        return None, None

    device = Pred_xy.device
    dtype = Pred_xy.dtype
    pred_m = torch.empty_like(Pred_xy, dtype=dtype)
    target_m = torch.empty_like(Target_xy, dtype=dtype)
    if 'x_grid_to_m' in bounds and 'y_grid_to_m' in bounds:
        ax, bx = bounds['x_grid_to_m']
        ay, by = bounds['y_grid_to_m']
        pred_m[..., 0] = Pred_xy[..., 0] * ax + bx
        pred_m[..., 1] = Pred_xy[..., 1] * ay + by
        target_m[..., 0] = Target_xy[..., 0] * ax + bx
        target_m[..., 1] = Target_xy[..., 1] * ay + by
    else:
        grid_max = float(bounds.get('grid_size', 1024) - 1)
        min_x = torch.tensor(bounds['min_x'], device=device, dtype=dtype)
        max_x = torch.tensor(bounds['max_x'], device=device, dtype=dtype)
        min_y = torch.tensor(bounds['min_y'], device=device, dtype=dtype)
        max_y = torch.tensor(bounds['max_y'], device=device, dtype=dtype)
        pred_m[..., 0] = Pred_xy[..., 0] / grid_max * (max_x - min_x) + min_x
        pred_m[..., 1] = Pred_xy[..., 1] / grid_max * (max_y - min_y) + min_y
        target_m[..., 0] = Target_xy[..., 0] / grid_max * (max_x - min_x) + min_x
        target_m[..., 1] = Target_xy[..., 1] / grid_max * (max_y - min_y) + min_y

    valid = (Target_xy[..., 0] != 0) if valid_mask is None else valid_mask.bool()
    dist = torch.sqrt(torch.pow(pred_m - target_m, 2).sum(-1))
    ADE_m = dist[valid].mean() if valid.any() else dist.new_zeros(())
    final_valid = valid[:, -1]
    FDE_m = dist[:, -1][final_valid].mean() if final_valid.any() else dist.new_zeros(())
    return ADE_m, FDE_m

def train_model(model, optimizer, criterion, scheduler, train_loader, test_loader, config):
    print("Training the model ...")
    patience_limit = config['patience_limit']
    sos = config['sos']
    eos = config['eos']
    train_loss, prev_average_loss, patience= [],  10000000.0, 0
    trainFDE, trainADE = [], []
    # Keep an independent snapshot of the best parameters.  Assigning
    # ``Best_Model = model`` would only keep a reference to the object that
    # continues to be updated in later epochs.
    Best_Model = copy.deepcopy(model)
    for epoch in range(config['epochs']):
        model.train()
        epoch_losses, epoch_ADE, epoch_FDE = [], [], []
        for Scene, Target, Adj_Mat_Scene in train_loader: # Scene & Taget => [B, SL0, Nnodes, Features], Adj_Mat_Scene => [B, SL, Nnodes, Nnodes]
            optimizer.zero_grad()
            target_valid = (
                Target[..., config['agent_valid_index']] > 0.5
                if config.get('mask_aware_inputs', False) else None
            )
            Scene, Scene_mask, Adj_Mat, Target = prep_model_input(Scene, Adj_Mat_Scene,Target, sos, eos, config)
            outputs = model(Scene, Scene_mask, Adj_Mat)
            if target_valid is None:
                loss = criterion(outputs.reshape(-1, 1024), Target.reshape(-1).long())
            else:
                logits = outputs[:, 1:-1]
                labels = Target[:, 1:-1]
                point_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1).long(),
                    reduction='none',
                ).reshape_as(labels)
                point_mask = target_valid.unsqueeze(-1).expand_as(labels)
                loss = (point_loss * point_mask).sum() / point_mask.sum().clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            optimizer.step()
            with torch.no_grad():
                _, ADE, FDE = Find_topk_selected_words(
                    outputs[:,1:-1], Target[:,1:-1],
                    None if target_valid is None else target_valid,
                )
            epoch_losses.append(loss.item())
            epoch_ADE.append(ADE)
            epoch_FDE.append(FDE)

        trainADE.append(sum(epoch_ADE)/ len(train_loader))
        trainFDE.append(sum(epoch_FDE)/ len(train_loader))
        train_loss.append(sum(epoch_losses) / len(train_loader))
        scheduler.step()

        #改：语法错误
        # log = f'Epoch [{epoch+1}/{config['epochs']}], Loss: {train_loss[-1]:.2f}, ADE: {trainADE[-1]:.3f}, FDE: {trainFDE[-1]:.2f}'
        log = (
            f"Epoch [{epoch + 1}/{config['epochs']}], "
            f"Loss: {train_loss[-1]:.2f}, "
            f"ADE: {trainADE[-1]:.3f}, "
            f"FDE: {trainFDE[-1]:.2f}"
        )
        savelog(log, config['ct'])
        # Saving the best model to the file
        if train_loss[-1] < prev_average_loss: # checkpoint update
            prev_average_loss = train_loss[-1]  # Update previous average loss
            patience = 0
            Best_Model = copy.deepcopy(model)
        elif patience > patience_limit:
                savelog(f'early stopping, Patience lvl1 , lvl2 {patience}', config['ct'])
                break
        patience += 1
        if config['Test_during_training'] and epoch % 5 == 3:
            _, ADE, FDE = test_model(model, test_loader, config)
            savelog(f"During Training, Test ADE: {ADE :.2f}, FDE: {FDE :.2f}", config['ct'])
            model.train()
            # print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.5f} MB")
            # print(f"Reserved memory:  {torch.cuda.memory_reserved() / 1024**2:.5f} MB")
    return Best_Model, train_loss, trainADE, trainFDE


def test_model(model, test_loader, config):
    sos = config['sos']
    eos = config['eos']
    savelog("Starting testing phase", config['ct'])
    model.eval()
    ade_sum, fde_sum, ade_count, fde_count, test_size = 0, 0, 0, 0, 0
    ade_sum_m, fde_sum_m, ade_count_m, fde_count_m = 0, 0, 0, 0
    has_metric = bool(config.get('metric_coord_bounds'))
    with torch.no_grad():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for Scene, Target, Adj_Mat_Scene in test_loader:
            target_valid = (
                Target[..., config['agent_valid_index']] > 0.5
                if config.get('mask_aware_inputs', False) else None
            )
            Scene, Scene_mask, Adj_Mat, Target = prep_model_input(Scene, Adj_Mat_Scene,Target, sos, eos, config)
            Pred_target = model(Scene, Scene_mask,Adj_Mat)
            Target = Target[:,1:-1,:, config['xy_indx']]
            Topk_Selected_words, ADE, FDE = Find_topk_selected_words(Pred_target[:,1:-1], Target, target_valid)
            ADE_m, FDE_m = metric_errors_from_grid(Topk_Selected_words, Target, config, target_valid)
            valid = (Target[..., 0] != 0) if target_valid is None else target_valid
            final_valid = valid[:, -1]
            pred_dist = torch.sqrt(((Topk_Selected_words - Target) ** 2).sum(-1))
            test_size += Scene.size(0)
            ade_sum += pred_dist[valid].sum()
            ade_count += valid.sum()
            fde_sum += pred_dist[:, -1][final_valid].sum()
            fde_count += final_valid.sum()
            if ADE_m is not None:
                batch_ade_count_m = valid.sum()
                batch_fde_count_m = final_valid.sum()
                ade_sum_m += ADE_m * batch_ade_count_m
                ade_count_m += batch_ade_count_m
                fde_sum_m += FDE_m * batch_fde_count_m
                fde_count_m += batch_fde_count_m
        end_event.record()
        torch.cuda.synchronize()
        Avg_inference_time = start_event.elapsed_time(end_event)/test_size
        Avg_ADE = ade_sum / ade_count.clamp_min(1)
        Avg_FDE = fde_sum / fde_count.clamp_min(1)
        if has_metric:
            Avg_ADE_m = ade_sum_m / ade_count_m.clamp_min(1)
            Avg_FDE_m = fde_sum_m / fde_count_m.clamp_min(1)
            config['last_ADE_m'] = Avg_ADE_m.item()
            config['last_FDE_m'] = Avg_FDE_m.item()
            log= (
                f"ADE is : {Avg_ADE:.3f} grid ({Avg_ADE_m:.3f} m) \n "
                f"FDE is: {Avg_FDE:.3f} grid ({Avg_FDE_m:.3f} m) \n "
                f"Inference time: {1000*Avg_inference_time:.3f} ms"
            )
        else:
            log= f"ADE is : {Avg_ADE:.3f} px \n FDE is: {Avg_FDE:.3f} px \n Inference time: {1000*Avg_inference_time:.3f} ms"
        savelog(log, config['ct'])
        return Topk_Selected_words, Avg_ADE, Avg_FDE

def prep_model_input(Scene, Adj_Mat_Scene,Target, sos, eos, config):
    Scene = attach_sos_eos(Scene, sos, eos)
    if config.get('mask_aware_inputs', False):
        node_present = Scene[:, 1:-1, :, config['agent_valid_index']].amax(dim=1)
        Scene[:, 0, :, config['agent_valid_index']] = node_present
        Scene[:, -1, :, config['agent_valid_index']] = node_present
    Adj_Mat = torch.cat((torch.ones_like(Adj_Mat_Scene[:,:1]), Adj_Mat_Scene, torch.ones_like(Adj_Mat_Scene[:,:1])), dim=1)
    Scene_mask = create_src_mask(Scene, config.get('agent_valid_index'))
    Target = attach_sos_eos(Target[:,:,:, config['xy_indx']], sos[:, config['xy_indx']], eos[:,config['xy_indx']])
    return Scene, Scene_mask, Adj_Mat, Target
if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")
