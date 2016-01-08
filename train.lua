require 'torch'
require 'nn'
require 'nngraph'

-- local imports
local utils = require 'misc.utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Video Captioning model')
cmd:text()
cmd:text('Options')

-- model options 
cmd:option('-model','meanpool','type of model to use? meanpool|frames|frames_cnn')
cmd:option('-batch_size', 16, 'size of mini-batch')
cmd:option('-max_seqlen', 80, 'maximum sequence length during training. seqlen = vidlen + caplen and truncates the video if necessary')

-- data loading/pre-processing
cmd:option('-video_dir', '', 'directory where to read video data from')
cmd:option('-label_dir', '', 'directory where to read labels/captions from')
cmd:option('-vocab_file', '', 'path to vocabulary file')
cmd:option('-save_dir', '', 'directory where to save pre-processed data')

cmd:text()
cmd:option('-id', 1, 'an id identifying this run/job. used when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')

local opt = cmd:parse(arg)

require (opt.model .. '.DataLoader')
local loader = DataLoader(opt, utils)