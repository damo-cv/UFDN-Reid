from .train_normal import do_train as do_train_normal
from .train_normal_pd import do_train_pd

def do_train_config(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            num_gallery,
            start_epoch,     # add for using self trained model
            logger,
            local_rank,
            paras,
            nssd_weight_dict,
            num_classes=576):
    if cfg.MODEL.IF_PD == 'yes':
        do_train_pd(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            num_gallery,
            start_epoch,     # add for using self trained model
            logger,
            local_rank,
            nssd_weight_dict
        )
    else:
        do_train_normal(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            num_gallery,
            start_epoch,     # add for using self trained model
            logger,
            local_rank
        )