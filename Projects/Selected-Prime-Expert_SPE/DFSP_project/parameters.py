import argparse

parser = argparse.ArgumentParser()


YML_PATH = {
    "mit-states": './config/mit-states.yml',
    "mit-states_ours": './config/mit-states_ours.yml',
    "mit-states_ours_0.05": './config/mit-states_ours_0.05.yml',
    "mit_states_ours_MC_dropout_all_ep20": './config/mit_states_ours_MC_dropout_all_ep20.yml',
    "mit_states_org_MC_dropout_all_ep20": './config/mit_states_org_MC_dropout_all_ep20.yml',
    "ut-zappos": './config/ut-zappos.yml',
    "cgqa": './config/cgqa.yml',
    "ut-zappos_Monte_Dropout_all": './config/ut-zappos_Monte_Dropout_all.yml',
    "ut-zappos_0.05_Monte_Dropout": './config/ut-zappos_0.05_Monte_Dropout.yml',
    "ut-zappos_ours": './config/ut-zappos_ours.yml',
    "ut-zappos_ours_0.05": './config/ut-zappos_ours_0.05.yml',
    "ut-zappos_ours_MC_dropout_all": './config/ut-zappos_ours_MC_dropout_all.yml',
    "ut-zappos_ours_MC_dropout1": './config/ut-zappos_ours_MC_dropout1.yml',
    "ut-zappos_ours_MC_dropout2": './config/ut-zappos_ours_MC_dropout2.yml',
    "ut-zappos_ours_MC_dropout1_ep5": './config/ut-zappos_ours_MC_dropout1_ep5.yml',
    "ut-zappos_ours_MC_dropout2_ep5": './config/ut-zappos_ours_MC_dropout2_ep5.yml',   
    "ut-zappos_ours_MC_dropout_all_ep5": './config/ut-zappos_ours_MC_dropout_all_ep5.yml',
    "ut-zappos_ours_MC_dropout1_ep10": './config/ut-zappos_ours_MC_dropout1_ep10.yml',
    "ut-zappos_ours_MC_dropout2_ep10": './config/ut-zappos_ours_MC_dropout2_ep10.yml',   
    "ut-zappos_ours_MC_dropout_all_ep10": './config/ut-zappos_ours_MC_dropout_all_ep10.yml',
    "ut-zappos_ours_MC_dropout_all_ep20": './config/ut-zappos_ours_MC_dropout_all_ep20.yml',
    "ut-zappos_org_MC_dropout_all_ep20": './config/ut-zappos_org_MC_dropout_all_ep20.yml',
    "config1_mit-states": './config1/mit-states.yml',
    "config1_mit-states_ours": './config1/mit-states_ours.yml',
    "config1_mit-states_ours_0.05": './config1/mit-states_ours_0.05.yml',
    "config1_ut-zappos": './config1/ut-zappos.yml',
    "config1_ut-zappos_ours": './config1/ut-zappos_ours.yml',
    "config1_ut-zappos_ours_0.05": './config1/ut-zappos_ours_0.05.yml',
    "config1_ut-zappos_val": './config1/ut-zappos_val.yml',
    "config1_ut-zappos_ours_val": './config1/ut-zappos_ours_val.yml',
    "config1_ut-zappos_ours_0.05_val": './config1/ut-zappos_ours_0.05_val.yml',
    "config1_ut-zappos_val_no_fs": './config1/ut-zappos_val_no_fs.yml',
    "config1_ut-zappos_ours_val_no_fs": './config1/ut-zappos_ours_val_no_fs.yml',
    "config1_ut-zappos_ours_0.05_val_no_fs": './config1/ut-zappos_ours_0.05_val_no_fs.yml',
    "config_tem_mit-states": './config_tem/mit-states.yml',
    "config_tem_ut-zappos": './config_tem/ut-zappos.yml',
    "config1_mit-states_val_no_fs": './config1/mit-states_val_no_fs.yml',
    "config_tem_mit-states_train": './config_tem/mit-states_train.yml',
    "mit-states2": './config_seed/mit-states2.yml',
    "ut-zappos2": './config_seed/ut-zappos2.yml',
    "mit-states3": './config_seed/mit-states3.yml',
    "ut-zappos3": './config_seed/ut-zappos3.yml',
    "mit-states4": './config_seed/mit-states4.yml',
    "ut-zappos4": './config_seed/ut-zappos4.yml',
    "mit-states1": './config_seed/mit-states1.yml',
    "ut-zappos1": './config_seed/ut-zappos1.yml'
    
}



#model config
parser.add_argument("--lr", help="learning rate", type=float, default=5e-05)
parser.add_argument("--dataset", help="name of the dataset", type=str, default='mit-states')
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-05)
parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-L/14")
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
parser.add_argument("--epoch_start", help="start epoch", default=0, type=int)
parser.add_argument("--train_batch_size", help="train batch size", default=48, type=int)
parser.add_argument("--eval_batch_size", help="eval batch size", default=16, type=int)
parser.add_argument("--fusion", default="BiFusion", help="cross modal fusion method, choices = [BiFusion, txt2img, img2txt, NoFusion, DeCom]",)
parser.add_argument("--context_length", help="sets the context length of the clip model", default=8, type=int)
parser.add_argument("--attr_dropout", help="add dropout to attributes", type=float, default=0.3)
parser.add_argument("--save_path", help="save path", type=str)
parser.add_argument("--save_every_n", default=5, type=int, help="saves the model every n epochs")
parser.add_argument("--save_model", help="indicate if you want to save the model state dict()", action="store_true")
parser.add_argument("--load_model", default=None, help="load the trained model")
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument("--gradient_accumulation_steps", help="number of gradient accumulation steps", default=1, type=int)

parser.add_argument("--open_world", help="evaluate on open world setup", default= False)
parser.add_argument("--bias", help="eval bias", type=float, default=1e3)
parser.add_argument("--topk", help="eval topk", type=int, default=1)
parser.add_argument("--text_encoder_batch_size", help="batch size of the text encoder", default=16, type=int)
parser.add_argument('--threshold', type=float, help="optional threshold")
parser.add_argument('--threshold_trials', type=int, default=50, help="how many threshold values to try")
parser.add_argument("--phase", help="evaluate on open world setup", default= False)
parser.add_argument("--tem", help="evaluate on open world setup", default= None)
parser.add_argument("--weighted", help="weighted voting ", default= False)