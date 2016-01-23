require 'hdf5'
local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)

  local info = path.join(opt.save_dir, 'info.json')

  local prepro = false
  if not path.exists(info) then
    prepro = true
  else
    local curDataInfo = utils.read_json(info)
    if curDataInfo['max_seqlen'] ~= opt.max_seqlen then
      prepro = true
    end
  end

  if prepro then
    self:preprocess(opt.video_dir, opt.label_dir, opt.vocab_file, opt.save_dir, opt.max_seqlen)
  end

  self.batchSize = opt.batch_size
  self.labelsPerVid = opt.labels_per_vid
  self.max_seqlen = opt.max_seqlen
  print('loading preprocessed files...')
  local dataInfo = utils.read_json(info)

  self.word_to_ix = dataInfo['word_to_ix']
  self.ix_to_word = dataInfo['ix_to_word']
  self.vocabSize = 0
  for k,v in pairs(self.word_to_ix) do
    self.vocabSize = self.vocabSize + 1
  end 

  self.h5_train = hdf5.open(path.join(opt.save_dir, 'train.h5'), 'r')
  self.h5_val = hdf5.open(path.join(opt.save_dir, 'val.h5'), 'r')

  local numTrainLabels = self.h5_train:read('/labels'):dataspaceSize()[1]
  local numValLabels = self.h5_val:read('/labels'):dataspaceSize()[1]
  local numValVideos = self.h5_val:read('/videos'):dataspaceSize()[1]

  if numTrainLabels % self.batchSize ~= 0 then
    repeat
      numTrainLabels = numTrainLabels - 1
    until numTrainLabels % self.batchSize == 0
  end

  self.splits = {'train', 'val'}
  local numTrainVideos = self.h5_train:read('/videos'):dataspaceSize()[1]
  local numValVideos = self.h5_val:read('/videos'):dataspaceSize()[1]
  self.splitSizes = {numTrainVideos, numValVideos, numValVideos}

  self.data_files = {self.h5_train, self.h5_val}
  self.iter = {0, 0, 0}

  self.labelsPerVideo = {}
  self.vidCapIter = {}
  for splitIx=1, #self.splits do
    table.insert(self.labelsPerVideo, {})
    table.insert(self.vidCapIter, {})
    for k=1,self.splitSizes[splitIx] do
      local start_ix = self.data_files[splitIx]:read('label_start_ix'):partial({k,k})[1]
      local end_ix = self.data_files[splitIx]:read('label_end_ix'):partial({k,k})[1]
      table.insert(self.labelsPerVideo[splitIx], end_ix-start_ix+1)
      table.insert(self.vidCapIter[splitIx], 1)
    end
  end

  for i=1,#self.splits do   
    if self.splitSizes[i] % self.batchSize ~= 0 then
      repeat
        self.splitSizes[i] = self.splitSizes[i] - 1
      until self.splitSizes[i] % self.batchSize == 0
    end
    self.splitSizes[i] = self.splitSizes[i]/self.batchSize
  end
end

function DataLoader:getBatch(split_ix)

  if split_ix == 3 then 
    video, id = self:getEvalBatch()
    return video, nil, id
  end

  local batchSize

  -- find max vid size 
  local longestVid = 0
  local longestCap = 0
  for i=1,self.batchSize do -- find longest caption and video in batch
    local vidId = self.iter[split_ix]*self.batchSize + i
    local capStartIx = self.data_files[split_ix]:read('label_start_ix'):partial({vidId,vidId})[1]
    local capix = capStartIx + self.vidCapIter[split_ix][vidId] - 1
    for k=1,self.labelsPerVid do
      if capix > capStartIx + self.labelsPerVideo[split_ix][vidId] - 1 then
        capix = capStartIx
      end
      local capSize = self.data_files[split_ix]:read('/label_length'):partial({capix,capix})[1]
      if capSize > longestCap then longestCap = capSize end
      capix = capix + 1
    end

    local vidSize = self.data_files[split_ix]:read('/video_length'):partial({vidId, vidId})[1]
    if vidSize > longestVid then longestVid = vidSize end
  end

  local batchVideos = {}
  if longestVid + longestCap > self.max_seqlen then longestVid = self.max_seqlen - longestCap end
  for i=1,longestVid do table.insert(batchVideos, torch.ByteTensor(self.batchSize, 3, 256, 256)) end

  for i=1,self.batchSize do
    local vidId = self.iter[split_ix]*self.batchSize + i
    local vidlen = self.data_files[split_ix]:read('/video_length'):partial({vidId,vidId})[1]
    if vidlen > longestVid then vidlen = longestVid end
    for frameNum=1,vidlen do
      local frame = self.data_files[split_ix]:read('/videos'):partial({vidId,vidId}, {frameNum,frameNum}, {1,3}, {1,256}, {1,256})
      frame = frame:select(1,1):select(1,1)
      batchVideos[frameNum]:select(1, i):copy(frame)
    end
  end

  local batchLabels = torch.LongTensor(longestCap, self.batchSize*self.labelsPerVid):zero()
  local batchCapId = 0
  for i=1,self.batchSize do
    local vidId = self.iter[split_ix]*self.batchSize + i
    local capStartIx = self.data_files[split_ix]:read('label_start_ix'):partial({vidId,vidId})[1]
    for k=1,self.labelsPerVid do
      batchCapId = batchCapId + 1
      local capix = capStartIx + self.vidCapIter[split_ix][vidId] - 1
      if capix > capStartIx + self.labelsPerVideo[split_ix][vidId] - 1 then
        self.vidCapIter[split_ix][vidId] = 1
        capix = capStartIx + self.vidCapIter[split_ix][vidId] - 1
      end
      local caplen = self.data_files[split_ix]:read('/label_length'):partial({capix,capix})[1]
      local cap = self.data_files[split_ix]:read('/labels'):partial({capix,capix}, {1,caplen}):select(1,1)
      batchLabels:select(2,batchCapId):sub(1,cap:nElement()):copy(cap)
      
      self.vidCapIter[split_ix][vidId] = self.vidCapIter[split_ix][vidId] + 1
    end
  end
  self.iter[split_ix] = self.iter[split_ix] + 1

  if self.iter[split_ix] >= self.splitSizes[split_ix] then
    self.iter[split_ix] = 0
  end

  return batchVideos, batchLabels
end

function DataLoader:getEvalBatch()
  self.iter[3] = self.iter[3] + 1
  local vidId = self.iter[3]

  local video = {}
  local vidlen = self.data_files[2]:read('/video_length'):partial({vidId, vidId})[1]
  for i=1,vidlen do table.insert(video, torch.ByteTensor(1, 3, 256, 256)) end

  for frameNum=1,vidlen do
    local frame = self.data_files[2]:read('/videos'):partial({vidId, vidId}, {frameNum,frameNum}, {1,3}, {1,256}, {1,256})
    frame = frame:select(1,1):select(1,1)
    video[frameNum]:select(1, 1):copy(frame)
  end

  return video, 'vid' .. (vidId + 1200)
end

function DataLoader:getVocabSize()
  return self.vocabSize
end

function DataLoader:getVocab()
  return self.ix_to_word
end

function DataLoader:splitSize(split_ix)
  return self.splitSizes[split_ix]
end

function DataLoader:resetIterator(split_ix)
  self.iter[split_ix] = 0
end

function DataLoader:preprocess(video_dir, label_dir, vocab_file, save_dir, max_seqlen)
  local out = {}
  out['max_seqlen'] = max_seqlen
  local word_to_ix, ix_to_word = self:loadVocab(vocab_file)
  out['word_to_ix'] = word_to_ix
  out['ix_to_word'] = ix_to_word

  local dir_args = ' --video_dir ' .. video_dir .. ' --label_dir ' .. label_dir .. ' --vocab_file ' .. vocab_file .. ' --save_dir ' .. save_dir
  os.execute('python ./frames_cnn/prepo.py --split_name train --max_frames ' .. max_seqlen .. dir_args)
  os.execute('python ./frames_cnn/prepo.py --split_name val' .. dir_args)

  utils.write_json(path.join(save_dir, 'info.json'), out)
end

function DataLoader:loadVocab(vocab_file)

  local word_to_ix = {}
  local ix_to_word = {}

  local f = io.open(vocab_file)
  rawdata = f:read()
  local vocab_size = 0
  repeat 
    vocab_size = vocab_size+1
    word_to_ix[rawdata] = vocab_size
    ix_to_word[vocab_size] = rawdata
    rawdata = f:read()
  until not rawdata
  f:close()

  return word_to_ix, ix_to_word
end