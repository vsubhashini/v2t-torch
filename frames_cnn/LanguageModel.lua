require 'nn'
local LSTM = require 'misc.LSTM'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size', 4096)
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.max_seq_len = utils.getopt(opt, 'max_seqlen')
  local dropout = utils.getopt(opt, 'drop_prob', 0)
  self.gpuid = opt.gpuid

  -- create the core lstm network. note +1 for both the START and END tokens
  self.cnn = net_utils.build_cnn(opt)
  self.feat_expander = nn.FeatExpander()
  self.core = LSTM.lstm(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, dropout)
  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the LanguageModel')
  -- self.clones = torch.load('/scratch/cluster/vsub/ssayed/cv/clones.t7')
  -- self.lookup_tables = torch.load('/scratch/cluster/vsub/ssayed/cv/tables.t7')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  for t=2,self.max_seq_len do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
  end
  -- torch.save('/scratch/cluster/vsub/ssayed/cv/clones.t7', self.clones)
  -- torch.save('/scratch/cluster/vsub/ssayed/cv/tables.t7', self.lookup_tables)
end

function layer:getModulesList()
  return {self.core, self.lookup_table}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.lookup_tables) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
end

function layer:sample(input, opt)
  local vid = input[1]
  local sample_max = utils.getopt(sample_max, 'sample_max', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)

  local nsteps = self.max_seq_len
  local vid_length = #vid
  local batch_size = 1

  self:_createInitState(batch_size)
  local state = self.init_state

  -- we will write output predictions into tensor seq
  local output_seq = torch.LongTensor(nsteps-vid_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(nsteps-vid_length, batch_size)
  local logprobs -- logprobs predicted in last time step

  for t=1,nsteps do
    local xt, it, sampleLogprobs
    if t <= vid_length then
      xt = vid[t]
      if xt:nDimension() == 1 then xt = xt:view(1, -1) end
    elseif t == vid_length+1 then
      -- feed in the start tokens
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1) -- Nx1 vector of START tokens 
      xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors)
    else -- take predictions from previous timestep
      if sample_max == 1 then
        -- use argmax "sampling"
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      else
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
        else
          -- scale logprobs by temperature
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        it = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
        it = it:view(-1):long() -- and flatten indices for downstream processing
      end
      xt = self.lookup_table:forward(it)
    end

    if t > vid_length+1 then
      output_seq[t-(vid_length+1)] = it
      seqLogprobs[t-(vid_length+1)] = sampleLogprobs:view(-1):float()
    end

    local inputs = {xt,unpack(state)}
    local out = self.core:forward(inputs)
    logprobs = out[self.num_state+1] -- last element is the output vector
    state = {}
    for i=1,self.num_state do table.insert(state, out[i]) end
  end

  return output_seq, seqLogprobs
end

function layer:updateOutput(input)
  local vid = input[1]
  local output_seq = input[2]
  local labels_per_vid = input[3]

  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass

  self.vid_length = #vid
  local cap_length = output_seq:size(1)
  local batch_size = output_seq:size(2)
  local nsteps = self.vid_length + cap_length -- vid_length + 1 (START) + size(word_sequence) - 1 (exclude END Token)

  self.output:resize(nsteps, batch_size, self.vocab_size+1)
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}
  self.inputs = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0

  if nsteps > self.max_seq_len then nsteps = self.max_seq_len end
  print(self.vid_length)
  for t=1,nsteps do
    print(t)
    local xt
    -- encoding sequence 
    if t <= self.vid_length then
      xt = vid[t] -- NxK sized input
      xt = self.cnn:forward(net_utils.cnn_prepro(xt, false, self.gpuid))
      print(xt:type())
      xt = self.feat_expander:forward{xt, labels_per_vid}
      print(xt:type())
    -- if there is an encoding sequence and it has just ended, feed in the START tokens 
    elseif t == self.vid_length+1 then 
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1) -- Nx1 vector of START tokens 
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors)
    -- feed in the output sequence 
    else
      local it = output_seq[t-(self.vid_length+1)]:clone()
      --[[
        seq may contain zeros as null tokens, make sure we take them out to any arbitrary token
        that won't make lookup_table crash with an error.
        token #1 will do, arbitrarily. This will be ignored anyway
        because we will carefully set the loss to zero at these places
        in the criterion, so computation based on this value will be noop for the optimization.
      --]]
      it[torch.eq(it,0)] = 1
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_tables[t]:forward(it)
    end

    -- construct the inputs
    self.inputs[t] = {xt,unpack(self.state[t-1])}
    -- forward the network
    local out = self.clones[t]:forward(self.inputs[t])
    -- process the outputs
    self.output[t] = out[self.num_state+1] -- last element is the output vector
    self.state[t] = {} -- the rest is state
    for i=1,self.num_state do table.insert(self.state[t], out[i]) end
    self.tmax = t
  end

  return self.output
end

-- gradOutput is same size as logprobs 
function layer:updateGradInput(input, gradOutput)
  local dframes = {} -- grad on video frames

  -- go backwards and lets compute gradients
  local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, gradOutput[t])
    local dinputs = self.clones[t]:backward(self.inputs[t], dout)
    -- split the gradient to xt and to state
    local dxt = dinputs[1] -- first element is the input vector
    dstate[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs[k]) end
    
    -- continue backprop of xt
    if t <= self.vid_length then
      dframes[t] = dxt
    else
      local it = self.lookup_tables_inputs[t]
      self.lookup_tables[t]:backward(it, dxt) -- backprop into lookup table
    end
  end

  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
  self.gradInput = {dimgs, torch.Tensor()}
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

-- output is same as logprobs from language model 
function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros

  local nsteps,batch_size,Mp1 = input:size(1), input:size(2), input:size(3)
  local cap_length = seq:size(1)
  local vid_length = nsteps-cap_length

  local loss = 0
  local n = 0
  for b=1,batch_size do -- iterate over batch
    for t=1,cap_length do -- iterate over output sequence 
      -- fetch the index of the next token in the sequence
      local target_index = seq[{t,b}] 

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{t+vid_length,b,target_index }] -- log(p)
        self.gradInput[{ t+vid_length,b,target_index }] = -1
        n = n + 1
      end

    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(batch_size)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end