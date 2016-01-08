require 'torch'
require 'nn'
require 'nngraph'

-- local imports
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Video Captioning model')
cmd:text()
cmd:text('Options')
-- model options 
cmd:option('-model', 'meanpool', 'type of model to use? meanpool|frames|frames_cnn')
cmd:option('-num_layers', 2, 'number of layers in lstm')
cmd:option('-rnn_size', 512, 'size of the rnn in number of hidden nodes in each layer')
-- data loading/pre-processing
cmd:option('-video_dir', '', 'directory where to read video data from')
cmd:option('-label_dir', '', 'directory where to read labels/captions from')
cmd:option('-vocab_file', '', 'path to vocabulary file')
cmd:option('-save_dir', '', 'directory where to save pre-processed data')
-- General Optimization
cmd:option('-max_seqlen', 80, 'maximum sequence length during training. seqlen = vidlen + caplen and truncates the video if necessary')
cmd:option('-batch_size', 16, 'size of mini-batch')
cmd:option('-epochs', -1, 'max number of epochs to run for (-1 = run forever)')
cmd:option('-optim','rmsprop', 'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate', 4e-4,'learning rate')

cmd:option('-id', 1, 'an id identifying this run/job. used when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')
local opt = cmd:parse(arg)

-- basic torch initializations
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

require (opt.model .. '.DataLoader')
local loader = DataLoader(opt)
utils.setVocab(loader:getVocab())

require (opt.model .. '.LanguageModel')
opt.vocab_size = loader:getVocabSize()
protos = {}
protos.lm = nn.LanguageModel(opt)
protos.crit = nn.LanguageModelCriterion()

if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

local params, grad_params = protos.lm:getParameters()
print('total number of parameters in model: ', params:nElement())

print('creating thin models for checkpointing...')
local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias') 
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end 

protos.lm:createClones()

collectgarbage()