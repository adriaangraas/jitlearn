from dae import RealtimeTopToBottomTest
basedir = "/export/scratch2/adriaan/zond_fastdvdnet_training"

expt_type = RealtimeTopToBottomTest
expt = RealtimeTopToBottomTest()
# if resume:
#   loads the to-be-resumed model
# else:
#   reads all savepoints of the model
match_fnames, match_times = expt.find_checkpoints(
    basedir
)
print(match_fnames)

kwargs = {'parent_dir': basedir}
for batch, fname in match_fnames.items():
    # load model
    kwargs['fname'] = fname
    expt = expt_type.resume(**kwargs)

    time = match_times[batch]
    curr_num = (int(time // expt.time_per_proj)
                + expt.slice_num_start)
    ran = range(curr_num, curr_num + 1)
    pairs_eval = [(curr_num,
                   expt.get_pairs_replay(ran))]
    # else:
    #     if pairs_eval is None:  # dataset is recycled
    #         pairs_eval = [(batch, iter(expt.pairs_eval))]

expt.trainer.eval(
    expt.model,
    pairs,
    plot=True,
    max_len=None,
    accuracies=expt.eval_accuracies,
    summary_writer=None,
    plot_fn=plot_fn)
