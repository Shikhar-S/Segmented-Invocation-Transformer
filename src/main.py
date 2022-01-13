import os
import pathlib
import sys
from model.loss_functions import LabelSmoothingKLLoss
from model.position_ffn import ActivationFunction

from decoder.decoding_strategy import GNMTGlobalScorer
run_base_path = pathlib.Path(__file__).parent.absolute()
module_path = os.path.abspath(os.path.join(run_base_path,'clai/tellina-baseline/src'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)

import torch
import argparse
import pytorch_lightning as pl

from model.transformer import Transformer
from attribution.attributor import Attributor

from main_utils import check_path, format_time, get_logger, get_device, set_random_seed, init_weights, str2bool, get_property_dic, get_free_gpus
from reader.datamodule import DataModule
from model.embedding_layer import Embeddings, SegmentEmbeddings
from model.segment_encoder import SIT
from model.decoder import TransformerDecoder
from decoder.translator import Translator
from decoder.bash_generator import get_score
import pickle

def build_model(args,dm):
    src_vocab_sz = dm.src_tokenizer.vocab_len
    trg_vocab_sz = dm.trg_tokenizer.vocab_len
    embedding_dim  = args.d_model
    pad_idx = dm.src_tokenizer.pad

    assert ((dm.word_vectors is None) or (dm.word_vectors.shape[0] == src_vocab_sz))

    input_embedding_layer = SegmentEmbeddings(embedding_dim,\
                                        src_vocab_sz,\
                                        pad_idx,\
                                        position_encoding=True,\
                                        freeze_word_vecs=False,\
                                        word_vectors=dm.word_vectors,
                                        max_norm=args.embedding_max_norm)

    invocation_encoder = SIT(   args.d_model,\
                                args.encoder_heads,\
                                args.d_ff,\
                                args.dropout,
                                args.attention_dropout,\
                                args.encoder_layers,\
                                input_embedding_layer,\
                                pos_ffn_activation_fn=ActivationFunction.relu)

    output_embedding_layer = Embeddings(word_vec_size = embedding_dim,\
                                word_vocab_size = trg_vocab_sz,\
                                word_padding_idx = pad_idx,\
                                position_encoding=True,\
                                freeze_word_vecs = False,\
                                word_vectors = None)

    decoder = TransformerDecoder(args.decoder_layers,
                                args.d_model,
                                args.decoder_heads,
                                args.d_ff,
                                args.dropout,
                                args.attention_dropout,
                                embeddings = output_embedding_layer,
                                padding_idx = pad_idx,
                                pos_ffn_activation_fn=ActivationFunction.relu)

    model = Transformer(encoder = invocation_encoder,\
                        decoder = decoder,\
                        d_model = args.d_model,\
                        trg_tokenizer=dm.trg_tokenizer,\
                        device = args.device)
    return model

def set_translator(args,model,dm):
    max_trg_len = max(dm.training_data.max_trg_length,dm.validation_data.max_trg_length)+10
    global_scorer = GNMTGlobalScorer(alpha = args.length_penalty,length_penalty="wu")
    translator = Translator(model = model,
                            datamodule=dm,
                            decoding_strategy=args.decoding_strategy,
                            n_best=args.n_best,
                            min_length=2,
                            max_length=max_trg_len,
                            report_score=False,
                            beam_size=args.beam_size,
                            global_scorer=global_scorer)
    model.init_translator(translator)

def get_results(model, dataloader):
    import time
    result = {'truth':[],'prediction':[],'all_metric':[],'invocation_text':[],'phrasing':[],'model_loss':[],'example_metric':[],'model_score':[]}
    print('inside get result')
    with torch.no_grad():
        bidx = 0
        timestart = time.time()
        for batch in dataloader:
            bidx += 1
            if bidx%100==0:
                print(bidx,flush=True)
                curtime = time.time()
                time_per_batch = (curtime-timestart)/(bidx-1.0)
                total_min,total_sec = format_time(timestart,curtime)
                print(f'time average per batch: {time_per_batch} sec',flush=True)
                print(f'total time: {total_min} min {total_sec} sec',flush=True)

            (segment_batch, segment_len_batch),(y,y_len)= batch
            y = y.transpose(1,0)

            translations = model.translator.translate(segment_batch, segment_len_batch, y,y_len)
            truth =[[" ".join(t) for t in translation.truth] for translation in translations]
            pred=[[" ".join(p) for p in translation.pred] for translation in translations]
            model_score = [translation.pred_score for translation in translations]
            scores,all_scores = get_score(truth, pred)

            #compute loss
            output = model(sentence_phrases = segment_batch,\
                            phrases_len = segment_len_batch,\
                            target = y,\
                            target_len = y_len)
            loss_fn = LabelSmoothingKLLoss(label_smoothing=0.1,tgt_vocab_size=model.trg_tokenizer.vocab_len,ignore_index=model.pad,reduction='none')
            loss = loss_fn(output[:-1,:,:].permute(1,2,0),y[1:,:,0].transpose(1,0))
            loss = loss.sum(dim=-1) #batch size X seq len X classes
            loss = loss.detach().cpu().numpy() #batch size X seq len

            result['truth'].extend(truth)
            result['prediction'].extend(pred)
            result['invocation_text'].extend([" ".join(translation.inv) for translation in translations])
            result['phrasing'].extend([[" ".join(p) for p in translation.inv_phrasing] for translation in translations])

            result['all_metric'].extend(all_scores)
            result['example_metric'].extend(scores)
            result['model_score'].extend(model_score)
            result['model_loss'].extend([L for L in loss]) # List of numpy arrays

        final_min, final_sec = format_time(timestart,time.time())
        print(f'time final: {final_min} min {final_sec} sec',flush=True)

    return result

def get_model_and_dataloader(args):
    chkpt_file = os.listdir(args.checkpoint_path)[0]
    checkpoint_file_path = args.checkpoint_path + chkpt_file
    print('Split',args.split)
    print('Checkpoint_path',checkpoint_file_path)

    dm = DataModule(args)
    dm.setup()
    print('evaluation with subtree cut height=',dm.subtree_cut_height)

    model = build_model(args,dm)
    model_loaded = torch.load(checkpoint_file_path, map_location=args.device)
    model.load_state_dict(model_loaded['state_dict'])
    model.eval()
    model.freeze()
    set_translator(args, model, dm)
    return model,dm

def run_inference(model, dataloader):
    with torch.no_grad():
        for batch in dataloader:
            (phrase_batch, phrase_len_batch),(y,y_len)= batch
            translations = model.translator.translate(phrase_batch, phrase_len_batch)

def benchmark(args):
    import pickle
    import torch.utils.benchmark as benchmark
    setup_stmt = 'model,dataloader = get_model_and_dataloader(args)'
    stmt = 'run_inference(model,dataloader.test_dataloader())'
    for num_threads in [1,2,3,4,5,6]:
        T = benchmark.Timer(stmt=stmt,setup=setup_stmt,globals=globals(),num_threads=num_threads)
        num_times = 5
        result = T.timeit(num_times)
        print(f'{result}s')
        with open(f'./inference_measurements/inference_time.{num_threads}th.{args.split}.pkl','wb') as f:
            pickle.dump(result,f)


def evaluate(args):
    #Ensure this is called for single gpu, else dataloader duplicates some examples to prevent deadlock.
    print('on eval')
    chkpt_file = os.listdir(args.checkpoint_path)[0]
    args.checkpoint_path = args.checkpoint_path + chkpt_file
    print('Split',args.split)
    print('Checkpoint_path',args.checkpoint_path)

    dm = DataModule(args)
    dm.setup()
    print('evaluation with subtree cut height=',dm.subtree_cut_height)

    model = build_model(args,dm)
    model_loaded = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(model_loaded['state_dict'])
    model.eval()
    model.freeze()
    set_translator(args, model, dm)
    result_path = f'../run/predictions_with_score/{args.n_training_examples}/'
    check_path(result_path,exist_ok=True)

    # results = get_results( model, dm.val_dataloader())
    # with open(f'{result_path}result_val.{args.split}.pkl','wb') as f:
    #     pickle.dump(results,f)
    # print('Dumped val results')

    results = get_results(model,dm.test_dataloader())
    with open(f'{result_path}result_test.{args.split}.pkl','wb') as f:
        pickle.dump(results,f)
    print('Dumped test results')

    # results = get_results(model,dm.train_dataloader())
    # with open(f'{result_path}result_train.{args.split}.pkl','wb') as f:
    #     pickle.dump(results,f)
    # print('Dumped train results')

# def check_data_loader(args):
#     logger.info(f'Running Split:{args.split}')
#     logger.info(f'Random Seed:{args.seed}')

#     dm = DataModule(args)
#     dm.setup()
#     for batch in dm.train_dataloader():
#         (leaves,nodes,matrix), (cmd,cmd_len) = batch
#         print(leaves.shape)
#         print(nodes.shape)
#         print(leaves)
#         for l in leaves[0,:]:
#             print(dm.src_tokenizer.leaf.get_token(l.item()))
#         print(nodes)
#         for n in nodes[0,:]:
#             print(dm.src_tokenizer.node.get_token(n.item()))
#         print(matrix.shape)
#         for sm in matrix[0]:
#             print(sm)
#         print(cmd.shape)
#         print(cmd_len.shape)
#         break
#     assert False

def train(args):
    # check_data_loader(args)
    logger.info(f'Running Split:{args.split}')
    logger.info(f'Random Seed:{args.seed}')

    dm = DataModule(args)
    dm.setup()
    model = build_model(args,dm)

    if dm.word_vectors is not None:
        init_ignore = ['emb_luts']
    else:
        init_ignore = []

    init_weights(model, init_ignore, gain = args.gain)
    set_translator(args, model, dm)

    run_base_path = "./"
    # run_base_path = pathlib.Path(__file__).parent.parent.absolute()
    run_dir = os.path.join(run_base_path,args.run_base_address,'split.' + str(args.split) + '.' + str(args.n_training_examples))
    logger.info('Storing model at:: '+run_dir)
    #callbacks for logging/monitoring
    earlystop_callback = pl.callbacks.EarlyStopping(patience=40,verbose=True, monitor = 'metric/validation', mode = 'max')
    checkpoint_callback = pl.callbacks.ModelCheckpoint( monitor='metric/validation', mode='max',save_top_k=1)
    # tb_logging_callback = TensorboardLoggingCallback(dm,args) Logs to tensorboard embeddings for tsne visualization
    gpu_list = get_free_gpus(memory_req=8000)
    print('Running on GPUs', gpu_list)
    trainer = pl.Trainer.from_argparse_args(args, track_grad_norm=-1, gradient_clip_val = args.gradient_clip_val,\
                                            gpus = gpu_list,accelerator="ddp",replace_sampler_ddp=False,\
                                            default_root_dir=run_dir,\
                                            callbacks = [earlystop_callback, checkpoint_callback],\
                                            accumulate_grad_batches = args.accumulate_grad_batches)
    trainer.fit(model=model,datamodule=dm)

def attribute(args):
    args.checkpoint_path = f'../run/split.{args.split}.-1'

    attributor = Attributor(args)

    #Load Model
    chkpt_file = 'relu_dfs.ckpt'
    chkpt_file = os.path.join(args.checkpoint_path,chkpt_file)
    print('Checkpoint_path',chkpt_file)

    model = build_model(args,attributor.dm)
    model_loaded = torch.load(chkpt_file, map_location=args.device)
    model = model.to(args.device)
    print(model.device)
    model.load_state_dict(model_loaded['state_dict'])
    model.eval()
    model.freeze()
    set_translator(args, model, attributor.dm)

    attributor.init_attr(model)
    attributor.run_integrated_gradients(dataloader_type='test')
    attributor.save_results(filepath='../run/attribution/attr_5ksteps.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type = int,default=83)
    parser.add_argument('--mode',type=str , default='train')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--checkpoint_path',type=str,default='')
    parser.add_argument('--verbose',type=str2bool,default = False)
    parser.add_argument('--log_graph',type=str2bool,default=False)
    parser.add_argument('--log_embeddings',type=str2bool,default=False)
    parser.add_argument('--guidance_distribution',type=str2bool,default=False)
    parser.add_argument('--n_training_examples', type = int, default=-1)
    parser.add_argument('--gradient_clip_val',type=float,default=2.754)
    parser.add_argument('--gain',type=float,default=0.6)
    parser.add_argument('--multigpu',type=str2bool,default=False)
    parser = DataModule.add_model_specific_args(parser)
    parser = Transformer.add_model_specific_args(parser)

    args,unparsed = parser.parse_known_args()
    args.device = get_device(args)
    global logger
    logger = get_logger()
    logger.info('__INIT_PROCESS__')
    logger.info('Arguments::' + str(args))
    set_random_seed(args.seed,args.device == torch.device('cuda'))
    if len(unparsed)>0:
        print('Unparsed args: %s',unparsed)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'benchmark':
        benchmark(args)
    elif args.mode == 'attribution':
        attribute(args)
    else:
        raise ValueError('Unknown mode: '+args.mode)
