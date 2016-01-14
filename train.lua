require 'torch'
require 'nn'
require 'nngraph'

-- local imports
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Video Captioning model')
cmd:text()
cmd:text('Options')
-- model options 
cmd:option('-model', 'meanpool', 'type of model to use? meanpool|frames')
cmd:option('-num_layers', 2, 'number of layers in lstm')
cmd:option('-rnn_size', 512, 'size of the rnn in number of hidden nodes in each layer')
cmd:option('-drop_prob', 0, 'probability for dropout (0 = no dropout)')
-- data loading/pre-processing
cmd:option('-video_dir', '', 'directory where to read video data from')
cmd:option('-label_dir', '', 'directory where to read labels/captions from')
cmd:option('-vocab_file', '/scratch/cluster/vsub/ssayed/youtube_dataset/labels/vocabulary.txt', 'path to vocabulary file')
cmd:option('-save_dir', '/scratch/cluster/vsub/ssayed/youtube_dataset/mean_pooled', 'directory where to save/load pre-processed data')
-- general optimization
cmd:option('-max_seqlen', 80, 'maximum sequence length during training. seqlen = vidlen + caplen and truncates the video if necessary')
cmd:option('-batch_size', 16, 'size of mini-batch')
cmd:option('-epochs', -1, 'max number of epochs to run for (-1 = run forever)')
-- optimization learning
cmd:option('-optim','rmsprop', 'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate', 4e-4,'learning rate')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-decay_rate', 50000, 'decay rate')
-- printing updates and saving checkpointsc
cmd:option('-lang_metric','METEOR','metric to use for saving checkpoints METEOR|CIDEr|ROUGE_L')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_model_every', 200, 'how many steps/minibatches between loss and language evaluation on model')
cmd:option('-eval_val_loss', 0, 'evaluate and save validation loss during checkpoints (1 == yes, 0 == no)')
cmd:option('-save_model_dir', 'cv/','directory to save checkpoints and score/loss evaluations')
cmd:option('-save_model_name', 'frames_1','name of model')
-- misc
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
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-- create data loader
require (opt.model .. '.DataLoader')
local loader = DataLoader(opt)
utils.setVocab(loader:getVocab())
local batchVideos, batchLabels, _, labelsPerVideo = loader:getBatch(1)

-- create model 
require (opt.model .. '.LanguageModel')
opt.vocab_size = loader:getVocabSize()
protos = {}
protos.lm = nn.LanguageModel(opt)
protos.crit = nn.LanguageModelCriterion()

-- send model to gpu if necessary 
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

-- create clones for timesteps
protos.lm:createClones()

collectgarbage()

local function sampleSplit(split_ix)
  protos.lm:evaluate()
  loader:resetIterator(split_ix) 

  local numSamples = loader:splitSize(split_ix)
  local splitSamples = {}

  local vidIds = {}
  for ix=1,numSamples do
	  local video, _, id = loader:getBatch(split_ix)

	  if opt.gpuid >= 0 then video, _ = net_utils.send_to_gpu(video) end

    local sample, logprobs = protos.lm:sample(video, {sample_max=opt.sample_max, temperature=opt.temperature})

    table.insert(splitSamples, sample)
    table.insert(vidIds, id)
  end

  return splitSamples, vidIds
end

local function evalSplit(split_ix)
  protos.lm:evaluate()
  loader:resetIterator(split_ix) 

  local totalLoss = 0
  local numEvals = loader:splitSize(split_ix)

  for i=1,numEvals do

    -- fetch a batch of data
 		local batchVideos, batchLabels, _ = loader:getBatch(split_ix)

		if opt.gpuid >= 0 then batchVideos, batchLabels = net_utils.send_to_gpu(batchVideos, batchLabels) end

    -- forward the model to get loss
    local logprobs = protos.lm:forward{batchVideos, batchLabels}
    local loss = protos.crit:forward(logprobs, batchLabels)
    totalLoss = totalLoss + loss

    print(i .. '/' .. numEvals .. '... ' .. loss)
  end

  return totalLoss/numEvals
end

local function lossFun()
  protos.lm:training()
  grad_params:zero()

  -- get batch of data  
  local batchVideos, batchLabels = loader:getBatch(1)
  
  if opt.gpuid >= 0 then batchVideos, batchLabels = net_utils.send_to_gpu(batchVideos, batchLabels) end

  -- forward pass
  local logprobs = protos.lm:forward{batchVideos, batchLabels}
  local loss = protos.crit:forward(logprobs, batchLabels)

  -- backward pass
  local dlogprobs = protos.crit:backward(logprobs, batchLabels)
  local ddumpy = protos.lm:backward({batchVideos, batchVideos}, dlogprobs)

  local gradNorm = grad_params:norm()
  if gradNorm > 5 then
    grad_params:mul(5)
    grad_params:div(gradNorm)
  end

  local losses = { total_loss = loss }
  return losses
end

local update = torch.Tensor(params:size()):zero()
if opt.gpuid >= 0 then
  update = update:float():cuda()
end

local ix_to_word = loader:getVocab()
local loss0
local iter = 0
local optim_state = {}
local ntrain = loader:splitSize(1)
local best_score = 0
while true do
  iter = iter + 1

  -- eval loss/gradient 
  local epoch = iter / ntrain
  local losses = lossFun()

  -- decay learning rate 
  local learning_rate = opt.learning_rate
  if epoch > opt.decay_start and opt.decay_start >= 0 then
    local epochs_over_start = math.ceil(epoch - opt.decay_start)
    local decay_factor = math.pow(opt.decay_rate, epochs_over_start)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end

  update:zero()
  -- optimization step
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state, update)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate, update)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state, update)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state, update)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- save checkpoint based on language evaluation
  if (iter % opt.eval_model_every == 0 or (epoch >= opt.epochs and opt.epochs > 0)) then

    local loss
    if opt.eval_val_loss > 0 then loss = evalSplit(2) end
    local splitSamples, ids = sampleSplit(3)
    scores, samples = utils.lang_eval(splitSamples, ids)

    -- save the model if it performs better than ever
    if scores[opt.lang_metric] > best_score then
      local checkpoint_path = path.join(opt.save_model_dir, opt.save_model_name)

      local checkpoint_info = {}
      checkpoint_info.opt = opt
      checkpoint_info.epoch = epoch
      checkpoint_info.vocab = ix_to_word
      checkpoint_info.scores = scores
      checkpoint_info.samples = samples
      if opt.eval_val_loss > 0 then checkpoint_info.loss = loss end
      utils.write_json(checkpoint_path .. '.json', checkpoint_info)

      local save_protos = {}
      save_protos.lm = thin_lm
      torch.save(checkpoint_path .. '.t7', save_protos)

      best_score = scores[opt.lang_metric]
    end
  end

  if iter % opt.print_every == 0 then
    print(string.format("%d (epoch %.3f), train_loss = %6.8f, ratio = %6.4e", iter, epoch, losses.total_loss, update:norm()/params:norm()))
  end

  if epoch > opt.epochs and opt.epochs > 0 then
    break
  end
end